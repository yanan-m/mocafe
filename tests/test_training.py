import logging
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchvision import models

from mocafe import _models, _training


def _train(weights, epochs=2):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 3))
    g = torch.Generator().manual_seed(1)
    dataset = TensorDataset(torch.randn(32, 4, generator=g), torch.arange(32) % 3, weights)
    loader = _training.make_loader(dataset, batch_size=8, num_workers=0, shuffle=True, seed=0)
    _training.train(model, loader, epochs=epochs, lr=0.1, device=torch.device("cpu"))
    return model


def test_training_is_reproducible():
    a, b = _train(torch.ones(32)), _train(torch.ones(32))
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)


def test_loss_weights_scale_the_gradient():
    # With zero weights only weight decay acts, so the result differs from unit weights.
    zero, ones = _train(torch.zeros(32)), _train(torch.ones(32))
    assert not torch.allclose(zero[1].weight, ones[1].weight)


@pytest.mark.parametrize("backbone", _models.BACKBONES)
def test_load_model_reads_both_checkpoint_formats(tmp_path, backbone):
    state = _models.build_model(backbone, 3).state_dict()
    torch.save({"model": state}, tmp_path / "wrapped.pt")
    torch.save(state, tmp_path / "plain.pt")
    torch.save({"model": state, "backbone": backbone}, tmp_path / "tagged.pt")
    for name in ("wrapped.pt", "plain.pt", "tagged.pt"):
        model = _training.load_model(tmp_path / name)
        assert _models.backbone_of(model) == backbone
        head = _models.classifier(model)
        assert head.out_features == 3
        assert torch.equal(
            head.weight, state[("fc" if backbone == "resnet50" else "head") + ".weight"]
        )


def test_load_model_rejects_other_checkpoints(tmp_path):
    torch.save({"model": nn.Linear(2, 2).state_dict()}, tmp_path / "linear.pt")
    with pytest.raises(ValueError, match="ResNet-50"):
        _training.load_model(tmp_path / "linear.pt")

    state = _models.build_model("resnet50", 3).state_dict()
    torch.save({"model": state, "backbone": "swin_t"}, tmp_path / "mislabelled.pt")
    with pytest.raises(ValueError, match="declared"):
        _training.load_model(tmp_path / "mislabelled.pt")

    microsoft = {"patch_embed.proj.weight": torch.zeros(96, 3, 4, 4), "head.weight": torch.zeros(3)}
    torch.save({"model": microsoft}, tmp_path / "microsoft.pt")
    with pytest.raises(ValueError, match="torchvision"):
        _training.load_model(tmp_path / "microsoft.pt")


def test_resolve_device():
    assert _training.resolve_device("cpu") == torch.device("cpu")
    assert _training.resolve_device(None).type in ("cpu", "cuda")


def test_resnet_keeps_constant_sgd():
    optimizer, scheduler, clip = _training.make_optimizer(
        _models.build_model("resnet50", 3), 1e-5, 50, 10
    )
    assert isinstance(optimizer, torch.optim.SGD) and scheduler is None and clip is None
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-5 and group["momentum"] == 0.9 and group["weight_decay"] == 5e-4


def test_swin_uses_adamw_without_decay_on_norms_and_biases():
    model = _models.build_model("swin_t", 3)
    optimizer, scheduler, clip = _training.make_optimizer(model, 1e-5, 50, 10)
    assert isinstance(optimizer, torch.optim.AdamW) and clip == 5.0
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(p.ndim > 1 for p in decay["params"])
    assert "head.bias" in {names[id(p)] for p in no_decay["params"]}
    assert any(names[id(p)].endswith("relative_position_bias_table") for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(names)
    assert decay["lr"] == pytest.approx(1e-5 * _training.SWIN_WARMUP_START)


def test_warmup_cosine_schedule():
    factor = _training.warmup_cosine(total_steps=100, warmup_steps=20)
    assert factor(0) == pytest.approx(_training.SWIN_WARMUP_START)
    assert factor(10) == pytest.approx(0.5 + 0.5 * _training.SWIN_WARMUP_START)
    assert factor(20) == pytest.approx(1.0)
    mid = _training.SWIN_MIN_LR_RATIO + (1 - _training.SWIN_MIN_LR_RATIO) * 0.5
    assert factor(60) == pytest.approx(mid)
    assert factor(100) == pytest.approx(_training.SWIN_MIN_LR_RATIO)
    assert all(factor(s) >= factor(s + 1) for s in range(20, 100))
    assert math.isclose(factor(500), _training.SWIN_MIN_LR_RATIO)


def test_swin_training_follows_the_schedule(caplog):
    torch.manual_seed(0)
    model = models.SwinTransformer(
        patch_size=[4, 4],
        embed_dim=8,
        depths=[1, 1],
        num_heads=[1, 1],
        window_size=[2, 2],
        num_classes=3,
    )
    dataset = TensorDataset(torch.randn(32, 3, 16, 16), torch.arange(32) % 3, torch.ones(32))
    loader = _training.make_loader(dataset, batch_size=8, num_workers=0, shuffle=True, seed=0)
    epochs, steps = 7, 4
    with caplog.at_level(logging.INFO, logger="mocafe"):
        _training.train(model, loader, epochs=epochs, lr=1e-3, device=torch.device("cpu"))
    factor = _training.warmup_cosine(epochs * steps, _training.SWIN_WARMUP_EPOCHS * steps)
    logged = [float(r.getMessage().rsplit("lr ", 1)[1]) for r in caplog.records]
    expected = [1e-3 * factor(e * steps) for e in range(1, epochs + 1)]
    assert logged == pytest.approx(expected, rel=1e-2)
