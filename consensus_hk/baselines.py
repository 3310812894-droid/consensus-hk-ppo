"""Five reported synchronous HK baselines, without RL or archive dependencies.

The algorithms retain the prespecified project adaptations of the cited models;
scalar/Euclidean convergence guarantees are not transferred to mean-L1 opinions.
"""

from __future__ import annotations

import hashlib
import operator

import numpy as np

from .hk_dynamics import calculate_distance_matrix, hk_update
from .metrics import (
    calculate_final_oprr,
    calculate_gravity_shift,
    calculate_total_variance,
    strict_reversal_counts,
)


METHODS = ("hk", "scod", "adaptive_confidence", "noise_hk", "inertial")
METHOD_LABELS = {
    "hk": "Classical HK",
    "scod": "Set-based confidence (SCOD)",
    "adaptive_confidence": "Adaptive-confidence HK (multidimensional extension)",
    "noise_hk": "Noise-HK (multidimensional extension)",
    "inertial": "Inertial HK",
}
SOURCES = {
    "hk": {"status": "project HK control", "distance": "mean absolute coordinate difference"},
    "scod": {
        "doi": "10.1109/LCSYS.2024.3479275",
        "title": "Opinion Dynamics With Set-Based Confidence: Convergence Criteria and Periodic Solutions",
        "adaptation": "union of coordinatewise confidence slabs; full-vector averaging",
    },
    "adaptive_confidence": {
        "doi": "10.1137/23M1558951",
        "title": "Bounded-Confidence Models of Opinion Dynamics with Adaptive Confidence Bounds",
        "adaptation": "replace scalar absolute distance by project mean-L1 distance",
    },
    "noise_hk": {
        "doi": "10.1016/j.automatica.2017.08.008",
        "title": "Noise leads to quasi-consensus of Hegselmann-Krause opinion dynamics",
        "adaptation": "synchronous vector HK, independent coordinate noise, then clipping",
    },
    "inertial": {
        "url": "https://arxiv.org/abs/1502.03332",
        "title": "Inertial Hegselmann-Krause Systems",
        "adaptation": "retention 0.5; project mean-L1 neighborhood replaces Euclidean distance",
    },
}
ADAPTIVE_REINFORCEMENT = 0.1
ADAPTIVE_DECAY = 0.9
INERTIAL_RETENTION = 0.5
NOISE_RATIO = 0.25
NOISE_STREAM_LABEL = 0x4E484B


def _average_trusted(current: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    weights = np.asarray(trusted, dtype=np.float64).copy()
    np.fill_diagonal(weights, 1.0)
    return np.clip(weights @ current / weights.sum(axis=1, keepdims=True), 0.0, 1.0)


def _snapshot(initial: np.ndarray, current: np.ndarray) -> dict:
    valid, reversals, tied = strict_reversal_counts(initial, current)
    denominator = int(np.sum(valid))
    return {
        "variance": float(calculate_total_variance(current)),
        "gravity": float(calculate_gravity_shift(initial, current)),
        "oprr_percent": float(100.0 * np.sum(reversals) / denominator) if denominator else 0.0,
        "valid_pairs": denominator,
        "reversal_count": int(np.sum(reversals)),
        "terminal_tied_pairs": int(np.sum(tied)),
    }


def run_baseline(
    method: str,
    initial: np.ndarray,
    epsilon: float,
    seed: int,
    max_rounds: int = 200,
    variance_threshold: float = 0.05,
    gravity_threshold: float = 0.45,
    stop_on_success: bool = True,
    record_trajectory: bool = False,
) -> tuple[dict, list[dict] | None]:
    """Return ``(terminal_row, trajectory_or_None)`` for one initial matrix.

    Success means terminal V <= variance_threshold AND G <= gravity_threshold.
    Default runs stop on the first joint success, including round zero. With
    ``stop_on_success=False`` the full budget is used and success still describes
    the FINAL state, not an earlier hit. A variance-only hit never stops a run.
    Trajectory includes round zero and independent ndarray copies of opinions;
    callers should save those matrices as NPZ rather than JSON.
    """
    if method not in METHODS:
        raise ValueError(f"unknown baseline: {method!r}")
    epsilon = float(epsilon)
    variance_threshold, gravity_threshold = float(variance_threshold), float(gravity_threshold)
    max_rounds, seed = operator.index(max_rounds), operator.index(seed)
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be finite and lie in [0,1]")
    if max_rounds <= 0 or seed < 0:
        raise ValueError("max_rounds must be positive; seed must be nonnegative")
    if not all(np.isfinite(x) and x > 0 for x in (variance_threshold, gravity_threshold)):
        raise ValueError("terminal thresholds must be finite and positive")
    initial = np.asarray(initial, dtype=np.float64)
    if initial.ndim != 2 or min(initial.shape) <= 0 or not np.all(np.isfinite(initial)):
        raise ValueError("initial must be a nonempty, finite (N,M) opinion matrix")
    if np.any(initial < 0.0) or np.any(initial > 1.0):
        raise ValueError("initial opinions must lie in [0,1]")
    initial = initial.copy()
    current = initial.copy()
    n, m = initial.shape
    confidence = None
    if method == "adaptive_confidence":
        confidence = np.full((n, n), epsilon, dtype=np.float64)
        np.fill_diagonal(confidence, 0.0)
    dynamics_rng = None
    if method == "noise_hk":
        # A separate fixed stream: never consume the initial-state RNG or hash().
        dynamics_rng = np.random.default_rng(np.random.SeedSequence([seed, NOISE_STREAM_LABEL]))
    trajectory: list[dict] | None = [] if record_trajectory else None
    first_variance_round = first_variance_gravity = first_joint_success_round = None
    rounds = 0

    def observe(round_index: int) -> tuple[dict, bool]:
        nonlocal first_variance_round, first_variance_gravity, first_joint_success_round
        metrics = _snapshot(initial, current)
        variance_ok = metrics["variance"] <= variance_threshold
        gravity_ok = metrics["gravity"] <= gravity_threshold
        success = bool(variance_ok and gravity_ok)
        if first_variance_round is None and variance_ok:
            first_variance_round = float(round_index)
            first_variance_gravity = metrics["gravity"]
        if first_joint_success_round is None and success:
            first_joint_success_round = float(round_index)
        if trajectory is not None:
            trajectory.append({
                "method": method, "round": round_index, "equivalent_round": float(round_index),
                "intervention_steps": 0, **metrics,
                "variance_feasible": bool(variance_ok), "gravity_feasible": bool(gravity_ok),
                "joint_success": success, "opinions": current.copy(),
            })
        return metrics, success

    metrics, success = observe(0)
    if not (stop_on_success and success):
        for rounds in range(1, max_rounds + 1):
            if method == "scod":
                # The union condition selects whole vectors, not topicwise means.
                differences = np.abs(current[:, None, :] - current[None, :, :])
                current = _average_trusted(current, np.any(differences <= epsilon, axis=2))
            elif method == "adaptive_confidence":
                # Both updates use OLD U and C; other-agent trust is STRICT D < C.
                distances = calculate_distance_matrix(current)
                receptive = distances < confidence
                successor = _average_trusted(current, receptive)
                confidence = np.where(
                    receptive,
                    confidence + ADAPTIVE_REINFORCEMENT * (1.0 - confidence),
                    ADAPTIVE_DECAY * confidence,
                )
                np.fill_diagonal(confidence, 0.0)
                current = successor
            elif method == "inertial":
                neighbor_mean = hk_update(current, epsilon)
                current = np.clip(
                    INERTIAL_RETENTION * current + (1.0 - INERTIAL_RETENTION) * neighbor_mean,
                    0.0, 1.0,
                )
            else:
                current = hk_update(current, epsilon)
                if method == "noise_hk":
                    amplitude = epsilon * NOISE_RATIO
                    noise = dynamics_rng.uniform(-amplitude, amplitude, size=current.shape)
                    current = np.clip(current + noise, 0.0, 1.0)
            metrics, success = observe(rounds)
            if stop_on_success and success:
                break

    # The paper generators have no initial ties; verify shared metric consistency.
    if metrics["valid_pairs"] == n * m * (m - 1) // 2:
        if not np.isclose(metrics["oprr_percent"], calculate_final_oprr(initial, current), rtol=0, atol=1e-12):
            raise RuntimeError("shared OPRR and strict reversal counts disagree")
    row = {
        "method": method, "model": METHOD_LABELS[method], "epsilon": epsilon, "seed": seed,
        "N": n, "M": m,
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "initial_variance": float(calculate_total_variance(initial)),
        "joint_success": success, "success": success,
        "variance_feasible": bool(metrics["variance"] <= variance_threshold),
        "gravity_feasible": bool(metrics["gravity"] <= gravity_threshold),
        "completed_synchronous_updates": rounds, "individual_intervention_steps": 0,
        "remaining_micro_steps": 0, "equivalent_rounds": float(rounds),
        "successful_equivalent_rounds": float(rounds) if success else None,
        "successful_intervention_steps": None,
        "final_variance": metrics["variance"], "gravity_shift": metrics["gravity"],
        "final_oprr": metrics["oprr_percent"], "valid_pairs": metrics["valid_pairs"],
        "reversal_count": metrics["reversal_count"], "terminal_tied_pairs": metrics["terminal_tied_pairs"],
        "first_variance_round": first_variance_round, "first_variance_gravity": first_variance_gravity,
        "first_joint_success_round": first_joint_success_round,
        "stop_reason": "joint_success" if stop_on_success and success else "max_rounds",
        "stop_on_success": bool(stop_on_success),
    }
    return row, trajectory
