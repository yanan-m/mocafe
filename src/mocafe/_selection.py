"""Algorithm 1 of MoCaFe: filtered teacher, discrepancy sampling and importance weights.

Every function works on the logits of the source model over the unlabelled target frames,
``logits[N, C]``, together with the track id of every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ETA = 1e-3  # smoothing of the sampling distribution, Eq. (9)
NUM_BINS = 10  # equal-frequency uncertainty bins U_k, Eq. (5)
EPS_R = 1e-6  # stabiliser of the reliability weights, Eq. (5)
_EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-frame ``CE(labels, softmax(logits))``."""
    p = softmax(logits)[np.arange(len(labels)), labels]
    return -np.log(np.clip(p, _EPS, None))


def normalized_entropy(logits: np.ndarray) -> np.ndarray:
    """u = -(1 / log C) sum_c p_c log p_c, in [0, 1], Eq. (3)."""
    p = softmax(logits)
    return -(p * np.log(np.clip(p, _EPS, None))).sum(axis=1) / np.log(logits.shape[1])


def track_index(track_ids) -> tuple[np.ndarray, int]:
    """Map arbitrary track ids to contiguous integers; every track needs at least two frames."""
    _, inverse, counts = np.unique(np.asarray(track_ids), return_inverse=True, return_counts=True)
    if counts.min() < 2:
        raise ValueError("MoCaFe needs every track to contain at least two frames.")
    return inverse, len(counts)


def leave_one_out_residuals(logits: np.ndarray, tracks: np.ndarray, n_tracks: int) -> np.ndarray:
    """rho_{i,l} = (1/C) ||z_{i,l} - mean_{tau != l} z_{i,tau}||^2, Eq. (4)."""
    sums = np.zeros((n_tracks, logits.shape[1]))
    np.add.at(sums, tracks, logits)
    lengths = np.bincount(tracks, minlength=n_tracks)[tracks, None]
    consensus = (sums[tracks] - logits) / (lengths - 1)
    return ((logits - consensus) ** 2).mean(axis=1)


def reliability_weights(
    u: np.ndarray, rho: np.ndarray, num_bins: int = NUM_BINS, eps_r: float = EPS_R
) -> np.ndarray:
    """r = 1 / (sigma_k^2 + eps_r), sigma_k^2 the mean residual of the frame's bin, Eq. (5)."""
    edges = np.quantile(u, np.linspace(0, 1, num_bins + 1)[1:-1])
    bins = np.searchsorted(edges, u, side="right")
    variance = np.bincount(bins, weights=rho) / np.maximum(np.bincount(bins), 1)
    return 1.0 / (variance[bins] + eps_r)


def filtered_logits(
    logits: np.ndarray, tracks: np.ndarray, n_tracks: int, reliability: np.ndarray
) -> np.ndarray:
    """Reliability-weighted fusion mu of each track, broadcast to its frames, Eq. (6)."""
    num = np.zeros((n_tracks, logits.shape[1]))
    np.add.at(num, tracks, reliability[:, None] * logits)
    den = np.bincount(tracks, weights=reliability, minlength=n_tracks)[:, None]
    return (num / den)[tracks]


def filtered_teacher(logits: np.ndarray, track_ids, num_bins: int = NUM_BINS) -> np.ndarray:
    """Filtered logits ``mu[N, C]`` of every frame's track; the teacher label is ``argmax mu``."""
    tracks, n_tracks = track_index(track_ids)
    u = normalized_entropy(logits)
    rho = leave_one_out_residuals(logits, tracks, n_tracks)
    return filtered_logits(logits, tracks, n_tracks, reliability_weights(u, rho, num_bins))


def confidence_features(mu: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """[N, 2] = (max softmax of the filtered logits, max softmax of the frame logits)."""
    return np.stack([softmax(mu).max(axis=1), softmax(logits).max(axis=1)], axis=1)


@dataclass(frozen=True)
class Selection:
    indices: np.ndarray  # [m] sampled frame indices, X_m
    labels: np.ndarray  # [m] teacher labels of the sampled frames
    beta: np.ndarray  # [m] importance weights, mean 1
    features: np.ndarray  # [m, 2] confidence features, used by p-MoCaFe


def select_frames(
    logits: np.ndarray, track_ids, num_samples: int, seed: int, num_bins: int = NUM_BINS
) -> Selection:
    """Algorithm 1, lines 1-18: sample ``num_samples`` frames by their teacher discrepancy.

    q_i = (h_i + eta) / sum_j (h_j + eta) with h = CE(y^t, f(x)) (Eq. 9). Frames are drawn with
    replacement and weighted by beta = 1 / (N q), rescaled to mean 1 over the sample (Eq. 10).
    """
    mu = filtered_teacher(logits, track_ids, num_bins)
    teacher = mu.argmax(axis=1)
    h = cross_entropy(logits, teacher)
    q = (h + ETA) / (h + ETA).sum()
    indices = np.random.default_rng(seed).choice(len(q), size=num_samples, replace=True, p=q)
    beta = 1.0 / (len(q) * q[indices])
    features = confidence_features(mu[indices], logits[indices])
    return Selection(indices, teacher[indices], beta / beta.mean(), features)
