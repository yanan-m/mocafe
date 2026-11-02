"""Algorithm 2 of p-MoCaFe: importance weights calibrated on a labelled public set.

The public set is filtered exactly like the target data. For every public frame we record the
confidence features ``(max softmax(mu), max softmax(z))`` and the corruption targets ``(p, d)``:
``(0, 1)`` when the filtered teacher is correct, otherwise ``(1, CE(y^t, z) / CE(y, z))``. Each
sampled target frame gets ``(p_hat, d_hat)`` by averaging its k nearest public neighbours, and
its weight becomes ``min(beta / (1 + p_hat (d_hat - 1)), 1)`` (Eq. 11 with clipping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mocafe._selection import NUM_BINS, confidence_features, cross_entropy, filtered_teacher

CLIP = 1.0
_EPS = 1e-12


@dataclass(frozen=True)
class Calibration:
    features: np.ndarray  # [v, 2] confidence features
    p: np.ndarray  # [v] 1 if the teacher is wrong, else 0
    d: np.ndarray  # [v] distortion CE(y^t, z) / CE(y, z), 1 when the teacher is correct


def build_calibration(
    logits: np.ndarray, labels: np.ndarray, track_ids, num_bins: int = NUM_BINS
) -> Calibration:
    """Algorithm 2, lines 3-11: corruption targets of the labelled public frames."""
    labels = np.asarray(labels)
    mu = filtered_teacher(logits, track_ids, num_bins)
    teacher = mu.argmax(axis=1)
    wrong = teacher != labels
    d = np.ones(len(labels))
    d[wrong] = cross_entropy(logits[wrong], teacher[wrong]) / np.maximum(
        cross_entropy(logits[wrong], labels[wrong]), _EPS
    )
    return Calibration(confidence_features(mu, logits), wrong.astype(float), d)


def default_k(public_size: int) -> int:
    """k = ceil(sqrt(v) / 2) neighbours for a public set of v frames (45 for v = 8000)."""
    return math.ceil(math.sqrt(public_size) / 2)


def knn_estimates(
    queries: np.ndarray, calibration: Calibration, k: int, chunk: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``(p, d)`` of the k nearest public features (Euclidean, stable tie order)."""
    ref = calibration.features
    p_hat, d_hat = np.empty(len(queries)), np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        dist = ((q[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
        nn = np.argsort(dist, axis=1, kind="stable")[:, :k]
        p_hat[start : start + chunk] = calibration.p[nn].mean(axis=1)
        d_hat[start : start + chunk] = calibration.d[nn].mean(axis=1)
    return p_hat, d_hat


def calibrated_weights(beta: np.ndarray, p_hat: np.ndarray, d_hat: np.ndarray) -> np.ndarray:
    """Eq. (11) clipped at 1: ``min(beta / (1 + p_hat (d_hat - 1)), 1)``."""
    c = np.maximum(1.0 + p_hat * (d_hat - 1.0), _EPS)
    return np.minimum(beta / c, CLIP)
