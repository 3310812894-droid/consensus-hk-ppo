"""Synchronous Hegselmann--Krause dynamics with mean absolute distance."""

from __future__ import annotations

import numpy as np


def _as_opinion_matrix(opinions: np.ndarray) -> np.ndarray:
    matrix = np.asarray(opinions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("opinions must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("opinions must contain only finite values")
    return matrix


def calculate_distance_matrix(opinions: np.ndarray) -> np.ndarray:
    """Return ``D[i, j] = mean_k(abs(U[i, k] - U[j, k]))``."""
    matrix = _as_opinion_matrix(opinions)
    differences = np.abs(matrix[:, None, :] - matrix[None, :, :])
    return np.mean(differences, axis=2)


def calculate_adjacency_matrix(opinions: np.ndarray, epsilon: float) -> np.ndarray:
    """Return the bounded-confidence adjacency matrix ``1[D <= epsilon]``."""
    threshold = float(epsilon)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("epsilon must be finite and lie in the physical range [0, 1]")
    return (calculate_distance_matrix(opinions) <= threshold).astype(np.float64)


def hk_update(
    opinions: np.ndarray,
    epsilon: float,
    opinion_low: float = 0.0,
    opinion_high: float = 1.0,
) -> np.ndarray:
    """Update all experts synchronously from the same opinion matrix."""
    matrix = _as_opinion_matrix(opinions)
    adjacency = calculate_adjacency_matrix(matrix, epsilon)
    neighbor_counts = adjacency.sum(axis=1, keepdims=True)
    neighbor_counts[neighbor_counts == 0.0] = 1.0
    updated = adjacency @ matrix / neighbor_counts
    return np.clip(updated, float(opinion_low), float(opinion_high))
