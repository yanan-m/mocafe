import numpy as np
import pytest

from mocafe._calibration import (
    Calibration,
    build_calibration,
    calibrated_weights,
    default_k,
    knn_estimates,
)
from mocafe._selection import filtered_teacher, softmax


def test_default_k_matches_paper():
    assert default_k(8000) == 45
    assert [default_k(4000), default_k(800), default_k(1)] == [32, 15, 1]


def test_build_calibration_targets():
    # Track "a" is filtered to class 0 (correct), track "b" to class 0 (wrong, label 1).
    logits = np.log(np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.7, 0.3]]))
    labels = np.array([0, 0, 1, 1])
    track_ids = ["a", "a", "b", "b"]
    cal = build_calibration(logits, labels, track_ids, num_bins=1)

    np.testing.assert_array_equal(cal.p, [0, 0, 1, 1])
    np.testing.assert_allclose(cal.d[:2], 1.0)
    np.testing.assert_allclose(cal.d[2:], np.log([0.9, 0.7]) / np.log([0.1, 0.3]))
    mu = filtered_teacher(logits, track_ids, num_bins=1)
    np.testing.assert_allclose(cal.features[:, 0], softmax(mu).max(1))
    np.testing.assert_allclose(cal.features[:, 1], [0.8, 0.6, 0.9, 0.7])


def test_knn_matches_brute_force():
    rng = np.random.default_rng(0)
    cal = Calibration(rng.random((300, 2)), rng.integers(0, 2, 300).astype(float), rng.random(300))
    queries = rng.random((50, 2))
    p_hat, d_hat = knn_estimates(queries, cal, k=7, chunk=16)
    for i, q in enumerate(queries):
        nn = np.argsort(np.linalg.norm(cal.features - q, axis=1))[:7]
        assert p_hat[i] == pytest.approx(cal.p[nn].mean())
        assert d_hat[i] == pytest.approx(cal.d[nn].mean())


def test_knn_breaks_ties_by_public_order():
    cal = Calibration(np.zeros((4, 2)), np.array([1.0, 0.0, 0.0, 0.0]), np.ones(4))
    p_hat, _ = knn_estimates(np.zeros((1, 2)), cal, k=1)
    assert p_hat[0] == 1.0


def test_weights_follow_eq11_and_clip_at_one():
    beta = np.array([0.5, 0.5, 2.0, 0.5, 4.0])
    p_hat = np.array([0.0, 0.5, 0.0, 1.0, 0.5])
    d_hat = np.array([1.0, 3.0, 1.0, 0.0, 0.5])
    w = calibrated_weights(beta, p_hat, d_hat)
    # c = [1, 2, 1, 0 (floored), 0.75]
    np.testing.assert_allclose(w, [0.5, 0.25, 1.0, 1.0, 1.0])
    assert np.isfinite(w).all() and w.max() <= 1.0
