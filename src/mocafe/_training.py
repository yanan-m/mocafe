"""Crop classifier: data loading, importance-weighted training and evaluation.

ResNet-50 is trained with SGD at a constant learning rate (paper, Section 5.1). Swin-T uses
AdamW with a 5-epoch linear warmup (paper) followed by cosine decay, the standard Swin fine-tuning
recipe for the settings the paper leaves open.
"""

from __future__ import annotations

import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from mocafe import _models

MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
SWIN_WEIGHT_DECAY = 0.05
SWIN_BETAS = (0.9, 0.999)
SWIN_WARMUP_EPOCHS = 5
SWIN_WARMUP_START = 1e-3  # warmup starts at lr * 1e-3
SWIN_MIN_LR_RATIO = 1e-2  # cosine decays to lr * 1e-2
SWIN_CLIP_GRAD = 5.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

logger = logging.getLogger("mocafe")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class CropDataset(Dataset):
    """Object crops with a label and a per-sample loss weight."""

    def __init__(self, root, images, labels, weights=None, transform=EVAL_TRANSFORM):
        self.root = Path(root)
        self.images = list(images)
        self.labels = [int(y) for y in labels]
        self.weights = [1.0] * len(self.images) if weights is None else [float(w) for w in weights]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.root / self.images[idx]).convert("RGB")
        return self.transform(image), self.labels[idx], torch.tensor(self.weights[idx])


def make_loader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool = False, seed: int = 0
) -> DataLoader:
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # Some OpenMP runtimes pin every forked worker to the parent's first core.
        if cpus is not None:
            os.sched_setaffinity(0, cpus)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def load_model(checkpoint: str | Path) -> nn.Module:
    """ResNet-50 or Swin-T classifier from a checkpoint, see :mod:`mocafe._models`."""
    return _models.load_model(checkpoint)


def warmup_cosine(total_steps: int, warmup_steps: int):
    """Learning-rate factor of each optimizer step: linear warmup, then cosine decay."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return SWIN_WARMUP_START + (1 - SWIN_WARMUP_START) * step / warmup_steps
        t = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        return SWIN_MIN_LR_RATIO + (1 - SWIN_MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * t))

    return factor


def make_optimizer(model: nn.Module, lr: float, epochs: int, steps_per_epoch: int):
    """``(optimizer, per-step scheduler or None, gradient-norm clip or None)`` for ``model``."""
    if not isinstance(model, models.SwinTransformer):
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
        return optimizer, None, None
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        skip = param.ndim <= 1 or name.endswith("relative_position_bias_table")
        (no_decay if skip else decay).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": SWIN_WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=SWIN_BETAS,
    )
    schedule = warmup_cosine(epochs * steps_per_epoch, SWIN_WARMUP_EPOCHS * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler, SWIN_CLIP_GRAD


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Logits ``[N, C]`` in dataset order."""
    model.to(device).eval()
    logits = []
    for inputs, _, _ in loader:
        with _autocast(device):
            logits.append(model(inputs.to(device)).float().cpu())
    return torch.cat(logits).numpy()


def train(
    model: nn.Module, loader: DataLoader, *, epochs: int, lr: float, device: torch.device
) -> None:
    """Minimise the weighted cross-entropy (1/m) sum_i w_i CE(y^t_i, f(x_i)), Eqs. (10), (12).

    The optimizer follows :func:`make_optimizer`. Shuffling and augmentation are reseeded every
    epoch from the loader's seed.
    """
    model.to(device)
    optimizer, scheduler, clip = make_optimizer(model, lr, epochs, len(loader))
    base_seed = loader.generator.initial_seed()
    for epoch in range(1, epochs + 1):
        torch.manual_seed(base_seed + epoch)
        loader.generator.manual_seed(base_seed + epoch)
        model.train()
        total, seen = 0.0, 0
        for inputs, labels, weights in loader:
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            with _autocast(device):
                logits = model(inputs)
            loss = (weights * F.cross_entropy(logits.float(), labels, reduction="none")).mean()
            optimizer.zero_grad()
            loss.backward()
            if clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * labels.size(0)
            seen += labels.size(0)
        logger.info(
            "epoch %d/%d: weighted loss %.4f, lr %.3g",
            epoch,
            epochs,
            total / seen,
            optimizer.param_groups[0]["lr"],
        )
