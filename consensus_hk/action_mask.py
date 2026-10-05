"""Continuous-action scaling and boundary-direction masking."""

from __future__ import annotations

import numpy as np


def scale_continuous_action(action: np.ndarray, delta: float) -> np.ndarray:
    """Clip an action to ``[-1, 1]`` and scale by physical step ``delta``."""
    raw_action = np.asarray(action, dtype=np.float64)
    step_size = float(delta)
    if raw_action.ndim != 1:
        raise ValueError("action must be a one-dimensional vector")
    if not np.all(np.isfinite(raw_action)):
        raise ValueError("action must contain only finite values")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("delta must be a positive finite value")
    return np.clip(raw_action, -1.0, 1.0) * step_size


def apply_boundary_action_mask(
    opinion_vector: np.ndarray,
    physical_action: np.ndarray,
    lower_boundary: float = 0.05,
    upper_boundary: float = 0.95,
) -> np.ndarray:
    """Suppress actions pointing outward from a boundary region."""
    opinions = np.asarray(opinion_vector, dtype=np.float64)
    action = np.asarray(physical_action, dtype=np.float64)
    if opinions.ndim != 1 or opinions.shape != action.shape:
        raise ValueError("opinion_vector and physical_action must be matching vectors")
    if not 0.0 <= lower_boundary < upper_boundary <= 1.0:
        raise ValueError("boundaries must satisfy 0 <= lower < upper <= 1")
    masked = action.copy()
    masked[(opinions <= lower_boundary) & (masked < 0.0)] = 0.0
    masked[(opinions >= upper_boundary) & (masked > 0.0)] = 0.0
    return masked
