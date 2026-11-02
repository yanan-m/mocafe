import numpy as np
import pytest

from mocafe._selection import (
    ETA,
    confidence_features,
    cross_entropy,
    filtered_logits,
    filtered_teacher,
    leave_one_out_residuals,
    normalized_entropy,
    reliability_weights,
    select_frames,
    track_index,
)


def _synthetic_logits(seed=0, n_tracks=50, length=4, classes=5):
    rng = np.random.default_rng(seed)
    track_ids = np.repeat(np.arange(n_tracks), length)
    logits = rng.normal(size=(n_tracks * length, classes))
    logits[np.arange(len(logits)), track_ids % classes] += 2.0
    return logits, track_ids


def test_eta_is_the_paper_value():
    assert ETA == 1e-3


def test_normalized_entropy_bounds():
    u = normalized_entropy(np.array([[0.0, 0.0, 0.0, 0.0], [50.0, 0.0, 0.0, 0.0]]))
    assert u[0] == pytest.approx(1.0)
    assert u[1] == pytest.approx(0.0, abs=1e-6)


def test_track_index_rejects_singletons():
    with pytest.raises(ValueError, match="at least two frames"):
        track_index(["a", "a", "b"])


def test_leave_one_out_residuals_match_definition():
    logits = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 2.0], [0.0, 1.0], [0.0, 3.0]])
    tracks, n = track_index(["a", "a", "a", "b", "b"])
    rho = leave_one_out_residuals(logits, tracks, n)
    # Frame 0 against the mean of frames 1 and 2: (4, 1) -> diff (-3, -1) -> (9 + 1) / 2.
    assert rho[0] == pytest.approx(5.0)
    # Two-frame track: each frame is compared with the other one: diff (0, -2) -> 4 / 2.
    assert rho[3] == pytest.approx(2.0)
    assert rho[4] == pytest.approx(2.0)


def test_reliability_is_inverse_bin_variance():
    u = np.array([0.1, 0.2, 0.8, 0.9])
    rho = np.array([1.0, 3.0, 10.0, 30.0])
    r = reliability_weights(u, rho, num_bins=2, eps_r=0.0)
    np.testing.assert_allclose(r, [1 / 2, 1 / 2, 1 / 20, 1 / 20])


def test_filtered_logits_downweight_unreliable_frames():
    logits = np.array([[2.0, 0.0], [2.0, 0.0], [0.0, 10.0]])
    tracks, n = track_index(["a", "a", "a"])
    reliable = filtered_logits(logits, tracks, n, np.array([1.0, 1.0, 0.01]))
    uniform = filtered_logits(logits, tracks, n, np.ones(3))
    np.testing.assert_allclose(reliable[0], [4.0 / 2.01, 0.1 / 2.01])
    assert reliable.argmax(1).tolist() == [0, 0, 0]
    assert uniform.argmax(1).tolist() == [1, 1, 1]


def test_filtered_teacher_is_shared_within_track():
    logits, track_ids = _synthetic_logits()
    mu = filtered_teacher(logits, track_ids)
    for t in np.unique(track_ids):
        np.testing.assert_allclose(mu[track_ids == t], mu[track_ids == t][[0]].repeat(4, 0))


def test_cross_entropy_and_confidence_features():
    assert cross_entropy(np.log([[0.25, 0.75]]), np.array([0]))[0] == pytest.approx(np.log(4.0))
    mu, z = np.log([[0.7, 0.3]]), np.log([[0.4, 0.6]])
    np.testing.assert_allclose(confidence_features(mu, z), [[0.7, 0.6]])


def test_select_frames_follows_eq9_and_eq10():
    logits, track_ids = _synthetic_logits()
    s = select_frames(logits, track_ids, 64, seed=1)

    mu = filtered_teacher(logits, track_ids)
    teacher = mu.argmax(1)
    h = cross_entropy(logits, teacher)
    q = (h + ETA) / (h + ETA).sum()
    expected = np.random.default_rng(1).choice(len(q), size=64, replace=True, p=q)
    np.testing.assert_array_equal(s.indices, expected)
    np.testing.assert_array_equal(s.labels, teacher[expected])
    raw = 1.0 / (len(q) * q[expected])
    np.testing.assert_allclose(s.beta, raw / raw.mean())
    assert s.beta.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(s.features, confidence_features(mu, logits)[expected])
    # beta = pi / q makes the weighted loss an unbiased estimate of the uniform mean, Eq. (10).
    loss = np.arange(len(q), dtype=float)
    assert np.sum(q * (1.0 / (len(q) * q)) * loss) == pytest.approx(loss.mean())


def test_select_frames_is_deterministic_per_seed():
    logits, track_ids = _synthetic_logits()
    a, b = select_frames(logits, track_ids, 64, 3), select_frames(logits, track_ids, 64, 3)
    np.testing.assert_array_equal(a.indices, b.indices)
    assert not np.array_equal(a.indices, select_frames(logits, track_ids, 64, 4).indices)
