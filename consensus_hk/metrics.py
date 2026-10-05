"""Consensus variance, normalized gravity shift, and strict ordinal reversals."""

from __future__ import annotations

import numpy as np


def _matching_opinion_matrices(
    initial_opinions: np.ndarray, current_opinions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_opinions, dtype=np.float64)
    current = np.asarray(current_opinions, dtype=np.float64)
    if initial.ndim != 2 or initial.shape != current.shape:
        raise ValueError("initial_opinions and current_opinions must have the same (N, M) shape")
    if initial.shape[0] == 0 or initial.shape[1] == 0:
        raise ValueError("opinion matrices must be non-empty")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(current)):
        raise ValueError("opinion matrices must contain only finite values")
    return initial, current


def calculate_total_variance(opinions: np.ndarray) -> float:
    """Return the sum of population variances over alternatives."""
    matrix = np.asarray(opinions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("opinions must be a non-empty (N, M) matrix")
    return float(np.sum(np.var(matrix, axis=0)))


def calculate_gravity_vector(opinions: np.ndarray) -> np.ndarray:
    """Return the group mean opinion vector."""
    matrix = np.asarray(opinions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("opinions must be a non-empty (N, M) matrix")
    return np.mean(matrix, axis=0)


def calculate_gravity_shift(
    initial_opinions: np.ndarray, current_opinions: np.ndarray,
) -> float:
    """Return ``||mean(U) - mean(U0)||_2 / sqrt(M)``."""
    initial, current = _matching_opinion_matrices(initial_opinions, current_opinions)
    displacement = calculate_gravity_vector(current) - calculate_gravity_vector(initial)
    return float(np.linalg.norm(displacement, ord=2) / np.sqrt(initial.shape[1]))


def strict_reversal_counts(
    initial_opinions: np.ndarray, current_opinions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-expert ``(valid_pairs, reversal_count, tied_pairs)`` arrays.

    A valid pair has a strict initial ordering. A reversal requires the
    opposite strict current ordering. ``tied_pairs`` counts valid initial
    pairs that are currently tied; these are not reversals. Initially tied
    pairs are excluded from all three counts and from the OPRR denominator.
    """
    initial, current = _matching_opinion_matrices(initial_opinions, current_opinions)
    left, right = np.triu_indices(initial.shape[1], k=1)
    initial_difference = initial[:, left] - initial[:, right]
    current_difference = current[:, left] - current[:, right]
    valid = initial_difference != 0.0
    reversed_pairs = ((initial_difference > 0.0) & (current_difference < 0.0)) | (
        (initial_difference < 0.0) & (current_difference > 0.0)
    )
    tied = valid & (current_difference == 0.0)
    return valid.sum(axis=1), reversed_pairs.sum(axis=1), tied.sum(axis=1)


def count_individual_inversions(
    initial_preference: np.ndarray, current_preference: np.ndarray,
) -> int:
    """Count strict reversals for one expert relative to the initial vector."""
    initial = np.asarray(initial_preference, dtype=np.float64)
    current = np.asarray(current_preference, dtype=np.float64)
    if initial.ndim != 1 or initial.shape != current.shape:
        raise ValueError("preference vectors must be one-dimensional and have the same shape")
    inversions = 0
    for left in range(initial.size):
        for right in range(left + 1, initial.size):
            initial_difference = initial[left] - initial[right]
            current_difference = current[left] - current[right]
            if initial_difference * current_difference < 0.0:
                inversions += 1
    return inversions


def calculate_training_ordinal_cost(
    initial_opinions: np.ndarray, current_opinions: np.ndarray,
) -> float:
    """Return the mean inversion count per expert, including HK effects."""
    initial, current = _matching_opinion_matrices(initial_opinions, current_opinions)
    counts = [count_individual_inversions(initial[i], current[i]) for i in range(initial.shape[0])]
    return float(np.mean(counts))


def calculate_incremental_ordinal_cost(
    initial_opinions: np.ndarray,
    previous_opinions: np.ndarray,
    current_opinions: np.ndarray,
) -> float:
    """Return ``C(t) - C(t-1)`` relative to the same initial opinions."""
    previous_cost = calculate_training_ordinal_cost(initial_opinions, previous_opinions)
    current_cost = calculate_training_ordinal_cost(initial_opinions, current_opinions)
    return float(current_cost - previous_cost)


def calculate_final_oprr(
    initial_opinions: np.ndarray, final_opinions: np.ndarray,
) -> float:
    """Return strict reversals as a percentage of initially strict pairs."""
    valid, reversed_pairs, _ = strict_reversal_counts(initial_opinions, final_opinions)
    total_pairs = int(valid.sum())
    if total_pairs == 0:
        return 0.0
    return float(100.0 * int(reversed_pairs.sum()) / total_pairs)


def summarize_terminal_metrics(
    initial_opinions: np.ndarray, final_opinions: np.ndarray,
) -> dict[str, float]:
    """Return the terminal variance, gravity shift, and OPRR."""
    return {
        "final_variance": calculate_total_variance(final_opinions),
        "gravity_shift": calculate_gravity_shift(initial_opinions, final_opinions),
        "final_oprr": calculate_final_oprr(initial_opinions, final_opinions),
    }
