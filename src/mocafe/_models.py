"""Supported classifier backbones: torchvision ResNet-50 and Swin-T.

A checkpoint is ``{"model": state_dict}`` (optionally with ``"backbone"``) or a plain
``state_dict``. Without the ``"backbone"`` entry the architecture is inferred from the classifier
keys: ResNet-50 has ``fc``, Swin-T has ``head`` and ``features``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

RESNET50 = "resnet50"
SWIN_T = "swin_t"
BACKBONES = (RESNET50, SWIN_T)

_CLASSIFIER = {RESNET50: "fc", SWIN_T: "head"}
_SWIN_PATCH_EMBED = "features.0.0.weight"  # Conv2d(3, 96, kernel 4, stride 4)


def build_model(backbone: str, num_classes: int, imagenet_init: bool = False) -> nn.Module:
    """A ``backbone`` classifier with a ``num_classes``-way linear head."""
    if backbone == RESNET50:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if imagenet_init else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == SWIN_T:
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if imagenet_init else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone {backbone!r}; expected one of {BACKBONES}")
    return model


def backbone_of(model: nn.Module) -> str:
    if isinstance(model, models.ResNet):
        return RESNET50
    if isinstance(model, models.SwinTransformer):
        return SWIN_T
    raise ValueError(f"Unsupported model type {type(model).__name__}")


def classifier(model: nn.Module) -> nn.Linear:
    return getattr(model, _CLASSIFIER[backbone_of(model)])


def detect_backbone(state: dict) -> str:
    """Backbone of a ``state_dict`` from its classifier and stem keys."""
    if "fc.weight" in state and "conv1.weight" in state:
        return RESNET50
    if "head.weight" in state and _SWIN_PATCH_EMBED in state:
        return SWIN_T
    if "head.weight" in state and "patch_embed.proj.weight" in state:
        raise ValueError(
            "This looks like a Microsoft/timm Swin checkpoint; convert it to the torchvision "
            "swin_t layout first."
        )
    raise ValueError("Not a ResNet-50 or Swin-T classifier checkpoint")


def save_model(model: nn.Module, path: str | Path, **extra) -> None:
    torch.save({"model": model.state_dict(), "backbone": backbone_of(model), **extra}, path)


def load_model(checkpoint: str | Path) -> nn.Module:
    """Classifier from a checkpoint (``{"model": state_dict, ...}`` or a plain ``state_dict``)."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = raw.get("model", raw)
    try:
        backbone = detect_backbone(state)
    except ValueError as e:
        raise ValueError(f"{checkpoint}: {e}") from None
    if isinstance(raw.get("backbone"), str) and raw["backbone"] != backbone:
        raise ValueError(f"{checkpoint}: declared {raw['backbone']!r} but holds {backbone!r}")
    model = build_model(backbone, state[f"{_CLASSIFIER[backbone]}.weight"].shape[0])
    model.load_state_dict(state)
    return model
