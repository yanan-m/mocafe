"""End-to-end runs of the public API on a tiny synthetic crop dataset (CPU)."""

import csv
import inspect
import json
import shutil

import numpy as np
import pytest
import torch
from PIL import Image

import mocafe
from mocafe import _calibration, _models, _selection, _training

SETTINGS = {"num_samples": 8, "epochs": 1, "batch_size": 4, "num_workers": 0, "device": "cpu"}
HEAD = {"resnet50": "fc.weight", "swin_t": "head.weight"}


def _write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _crops(root, name, n_tracks, length, rng):
    rows = []
    for t in range(n_tracks):
        for f in range(length):
            image = f"crops/{name}/{t}_{f}.jpg"
            (root / image).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (24, 16, 3), dtype=np.uint8)).save(root / image)
            rows.append({"image": image, "track_id": f"{name}{t}", "class_id": t % 3})
    return rows


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("mocafe")
    data_dir = root / "data"
    rng = np.random.default_rng(0)
    _write(data_dir / "target_adapt.csv", _crops(data_dir, "pool", 6, 3, rng))
    _write(data_dir / "target_test.csv", _crops(data_dir, "test", 3, 2, rng))
    _write(data_dir / "public_calibration.csv", _crops(data_dir, "public", 5, 2, rng))
    for backbone in _models.BACKBONES:
        torch.manual_seed(0)
        model = _models.build_model(backbone, 3)
        torch.save({"model": model.state_dict()}, root / f"{backbone}.pt")
    shutil.copy(root / "resnet50.pt", root / "source.pt")
    return root, data_dir


@pytest.fixture
def spy(monkeypatch):
    """Record the sampled frames and the training set of every adaptation run."""
    calls = {"selection": [], "train": []}
    select, train = _selection.select_frames, _training.train

    def select_spy(*args, **kwargs):
        calls["selection"].append(select(*args, **kwargs))
        return calls["selection"][-1]

    def train_spy(model, loader, **kwargs):
        calls["train"].append(loader.dataset)
        return train(model, loader, **kwargs)

    monkeypatch.setattr(_selection, "select_frames", select_spy)
    monkeypatch.setattr(_training, "train", train_spy)
    return calls


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def test_public_api():
    assert mocafe.__all__ == ["mocafe", "p_mocafe"]
    for fn in (mocafe.mocafe, mocafe.p_mocafe):
        assert "eta" not in inspect.signature(fn).parameters
        assert inspect.signature(fn).parameters["lr"].default is None
    assert mocafe.api.DEFAULT_LR == {
        ("resnet50", "mocafe"): 1e-5,
        ("resnet50", "p-mocafe"): 2e-5,
        ("swin_t", "mocafe"): 1e-5,
        ("swin_t", "p-mocafe"): 1e-5,
    }


@pytest.mark.parametrize("backbone", _models.BACKBONES)
def test_both_methods_share_the_selection(setup, spy, tmp_path, backbone):
    root, data_dir = setup
    ckpt = root / f"{backbone}.pt"
    a = mocafe.mocafe(ckpt, data_dir, tmp_path / "a", seed=1, **SETTINGS)
    b = mocafe.p_mocafe(ckpt, data_dir, tmp_path / "b", seed=1, **SETTINGS)

    for out, result, method in ((tmp_path / "a", a, "mocafe"), (tmp_path / "b", b, "p-mocafe")):
        assert sorted(p.name for p in out.iterdir()) == ["metrics.json", "model.pt"]
        metrics = json.loads((out / "metrics.json").read_text())
        assert result == {**metrics, "model": str(out / "model.pt")}
        assert metrics["method"] == method and metrics["backbone"] == backbone
        assert metrics["lr"] == mocafe.api.DEFAULT_LR[backbone, method]
        assert metrics["eta"] == 1e-3
        assert 0 <= metrics["source_accuracy"] <= 1 and 0 <= metrics["accuracy"] <= 1
        saved = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
        assert saved["backbone"] == backbone and HEAD[backbone] in saved["model"]
    assert b["k"] == _calibration.default_k(10) and b["public_frames"] == 10
    assert a["source_accuracy"] == b["source_accuracy"]

    sel_a, sel_b = spy["selection"]
    np.testing.assert_array_equal(sel_a.indices, sel_b.indices)
    np.testing.assert_array_equal(sel_a.beta, sel_b.beta)
    train_a, train_b = spy["train"]
    assert train_a.images == train_b.images and train_a.labels == train_b.labels
    np.testing.assert_allclose(train_a.weights, sel_a.beta)
    assert max(train_b.weights) <= 1.0
    assert not np.allclose(train_a.weights, train_b.weights)


@pytest.mark.parametrize("backbone", _models.BACKBONES)
def test_runs_are_reproducible(setup, tmp_path, backbone):
    root, data_dir = setup
    ckpt = root / f"{backbone}.pt"
    a = mocafe.p_mocafe(ckpt, data_dir, tmp_path / "a", **SETTINGS)
    b = mocafe.p_mocafe(ckpt, data_dir, tmp_path / "b", **SETTINGS)
    assert a["accuracy"] == b["accuracy"]
    state_a, state_b = _state(a["model"]), _state(b["model"])
    source = _state(ckpt)
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    assert not torch.equal(state_a[HEAD[backbone]], source[HEAD[backbone]])


def test_target_pool_labels_are_not_used(setup, tmp_path):
    root, data_dir = setup
    unlabelled = tmp_path / "data"
    shutil.copytree(data_dir, unlabelled)
    with open(data_dir / "target_adapt.csv", newline="") as f:
        rows = [{"image": r["image"], "track_id": r["track_id"]} for r in csv.DictReader(f)]
    _write(unlabelled / "target_adapt.csv", rows)

    a = mocafe.mocafe(root / "source.pt", data_dir, tmp_path / "a", **SETTINGS)
    b = mocafe.mocafe(root / "source.pt", unlabelled, tmp_path / "b", **SETTINGS)
    state_a, state_b = _state(a["model"]), _state(b["model"])
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)


@pytest.mark.parametrize("name", ["num_samples", "epochs", "batch_size", "lr"])
def test_rejects_non_positive_settings(setup, tmp_path, name):
    root, data_dir = setup
    with pytest.raises(ValueError, match=name):
        mocafe.mocafe(root / "source.pt", data_dir, tmp_path, **{**SETTINGS, name: 0})


def test_rejects_invalid_data(setup, tmp_path):
    root, data_dir = setup
    with pytest.raises(FileNotFoundError):
        mocafe.mocafe(root / "source.pt", tmp_path / "missing", tmp_path / "out", **SETTINGS)

    bad = tmp_path / "data"
    shutil.copytree(data_dir, bad)
    _write(bad / "public_calibration.csv", [{"image": "x.jpg", "class_id": 0}])
    with pytest.raises(ValueError, match="track_id"):
        mocafe.p_mocafe(root / "source.pt", bad, tmp_path / "out", **SETTINGS)

    shutil.copy(data_dir / "public_calibration.csv", bad / "public_calibration.csv")
    with open(data_dir / "target_test.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    _write(bad / "target_test.csv", [{**rows[0], "class_id": 7}, *rows[1:]])
    with pytest.raises(ValueError, match="class_id"):
        mocafe.mocafe(root / "source.pt", bad, tmp_path / "out", **SETTINGS)

    shutil.copy(data_dir / "target_test.csv", bad / "target_test.csv")
    with open(data_dir / "target_adapt.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    _write(bad / "target_adapt.csv", [{**rows[0], "track_id": "alone"}, *rows[1:]])
    with pytest.raises(ValueError, match="at least two frames"):
        mocafe.mocafe(root / "source.pt", bad, tmp_path / "out", **SETTINGS)
