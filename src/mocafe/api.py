"""End-to-end MoCaFe and p-MoCaFe adaptation of a source ResNet-50 or Swin-T classifier.

Both functions read the prepared crops and manifests in ``data_dir``:

- ``target_adapt.csv``: the unlabelled target frames (``image``, ``track_id``);
- ``target_test.csv``: the labelled target test frames (``image``, ``class_id``);
- ``public_calibration.csv``: the labelled public frames, p-MoCaFe only
  (``image``, ``track_id``, ``class_id``).

Image paths are relative to ``data_dir``. Target-pool labels are never read. The test set is
evaluated once before and once after adaptation, and is never used to choose a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from mocafe import _calibration, _data, _models, _selection, _training

logger = logging.getLogger("mocafe")

# Default learning rate per (backbone, method).
DEFAULT_LR = {
    (_models.RESNET50, "mocafe"): 1e-5,
    (_models.RESNET50, "p-mocafe"): 2e-5,
    (_models.SWIN_T, "mocafe"): 1e-5,
    (_models.SWIN_T, "p-mocafe"): 1e-5,
}


def mocafe(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    num_samples: int = 32000,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float | None = None,
    seed: int = 0,
    num_workers: int = 8,
    device: str | torch.device | None = None,
) -> dict:
    """Adapt a source model with MoCaFe (Algorithm 1).

    The source model labels every target track with its filtered teacher, ``num_samples``
    frames are drawn in proportion to their discrepancy with the teacher (eta = 0.001), and the
    model is fine-tuned on them with importance weights ``beta``.

    Args:
        checkpoint: Source ResNet-50 or Swin-T checkpoint (``{"model": state_dict}`` or a
            ``state_dict``); the backbone is detected from its keys.
        data_dir: Directory with the crops and the manifests listed in the module docstring.
        output_dir: Receives ``model.pt`` (the final adapted weights) and ``metrics.json``.
        num_samples: Number of sampled target frames ``m``.
        epochs: Fine-tuning epochs over the sampled frames.
        batch_size: Batch size for inference and fine-tuning.
        lr: Learning rate; ``None`` selects 1e-5. ResNet-50 uses SGD (momentum 0.9, weight
            decay 5e-4); Swin-T uses AdamW (weight decay 0.05) with a 5-epoch warmup and
            cosine decay.
        seed: Seed of the frame sampling, shuffling and augmentation.
        num_workers: DataLoader workers; augmentation streams depend on it.
        device: Torch device; defaults to CUDA when available.

    Returns:
        The contents of ``metrics.json`` plus the path of ``model.pt``.
    """
    return _adapt(
        "mocafe",
        checkpoint,
        data_dir,
        output_dir,
        num_samples,
        epochs,
        batch_size,
        lr,
        seed,
        num_workers,
        device,
    )


def p_mocafe(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    num_samples: int = 32000,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float | None = None,
    seed: int = 0,
    num_workers: int = 8,
    device: str | torch.device | None = None,
) -> dict:
    """Adapt a source model with p-MoCaFe (Algorithm 2).

    Uses the same teacher and sampled frames as :func:`mocafe` for the same seed. Each frame's
    weight becomes ``min(beta / (1 + p_hat (d_hat - 1)), 1)``, where ``(p_hat, d_hat)`` estimate
    the probability and distortion of a wrong pseudo-label from the ``ceil(sqrt(v) / 2)``
    nearest of the ``v`` labelled public frames in confidence-feature space.

    Arguments and return value are those of :func:`mocafe`; ``lr=None`` selects 2e-5 for
    ResNet-50 and 1e-5 for Swin-T.
    """
    return _adapt(
        "p-mocafe",
        checkpoint,
        data_dir,
        output_dir,
        num_samples,
        epochs,
        batch_size,
        lr,
        seed,
        num_workers,
        device,
    )


def _adapt(
    method: str,
    checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    num_samples: int,
    epochs: int,
    batch_size: int,
    lr: float | None,
    seed: int,
    num_workers: int,
    device: str | torch.device | None,
) -> dict:
    for name, value in (
        ("num_samples", num_samples),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("lr", 1.0 if lr is None else lr),
    ):
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers!r}")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    device = _training.resolve_device(device)
    pool = _data.read_manifest(data_dir / _data.TARGET_ADAPT, ("image", "track_id"))
    test = _data.read_manifest(data_dir / _data.TARGET_TEST, ("image", "class_id"))
    public = None
    if method == "p-mocafe":
        public = _data.read_manifest(data_dir / _data.PUBLIC, ("image", "track_id", "class_id"))

    _training.seed_everything(seed)
    model = _training.load_model(checkpoint)
    backbone = _models.backbone_of(model)
    if lr is None:
        lr = DEFAULT_LR[backbone, method]
    num_classes = _models.classifier(model).out_features
    test_labels = np.array(_data.class_ids(test, num_classes, data_dir / _data.TARGET_TEST))

    def logits_of(rows, labels=None):
        labels = [0] * len(rows) if labels is None else labels
        dataset = _training.CropDataset(data_dir, [r["image"] for r in rows], labels)
        loader = _training.make_loader(dataset, batch_size, num_workers, seed=seed)
        return _training.predict(model, loader, device)

    logger.info(
        "%s (%s): selecting %d of %d target frames", method, backbone, num_samples, len(pool)
    )
    selection = _selection.select_frames(
        logits_of(pool), [r["track_id"] for r in pool], num_samples, seed
    )
    weights = selection.beta
    extra = {}
    if public is not None:
        labels = _data.class_ids(public, num_classes, data_dir / _data.PUBLIC)
        calibration = _calibration.build_calibration(
            logits_of(public, labels), labels, [r["track_id"] for r in public]
        )
        k = _calibration.default_k(len(public))
        p_hat, d_hat = _calibration.knn_estimates(selection.features, calibration, k)
        weights = _calibration.calibrated_weights(selection.beta, p_hat, d_hat)
        extra = {"public_frames": len(public), "k": k}
    source_accuracy = float((logits_of(test).argmax(1) == test_labels).mean())

    train_set = _training.CropDataset(
        data_dir,
        [pool[i]["image"] for i in selection.indices],
        selection.labels,
        weights=weights,
        transform=_training.TRAIN_TRANSFORM,
    )
    loader = _training.make_loader(train_set, batch_size, num_workers, shuffle=True, seed=seed)
    _training.train(model, loader, epochs=epochs, lr=lr, device=device)
    accuracy = float((logits_of(test).argmax(1) == test_labels).mean())
    logger.info("%s: target test accuracy %.4f -> %.4f", method, source_accuracy, accuracy)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    _models.save_model(model, model_path)
    metrics = {
        "method": method,
        "backbone": backbone,
        "checkpoint": str(checkpoint),
        "eta": _selection.ETA,
        "num_samples": num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        **extra,
        "source_accuracy": source_accuracy,
        "accuracy": accuracy,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {**metrics, "model": str(model_path)}
