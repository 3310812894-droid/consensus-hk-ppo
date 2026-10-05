"""Initial-state generators used by the reported ID and four OOD cases."""

from __future__ import annotations

import operator

import numpy as np


OOD_DISTRIBUTIONS = ("beta_center", "beta_extremes", "beta_high", "bimodal")
AVAILABLE_DISTRIBUTIONS = ("uniform_control",) + OOD_DISTRIBUTIONS
DISTRIBUTION_DEFINITIONS = {
    "uniform_control": "iid Uniform(0,1)",
    "beta_center": "iid Beta(2,2)",
    "beta_extremes": "iid Beta(0.5,0.5)",
    "beta_high": "iid Beta(5,2)",
    "bimodal": "floor(N/2) experts Beta(2,8), remaining experts Beta(8,2); random row permutation",
}


def sample_initial_opinions(
    distribution: str,
    seed: int,
    n: int = 100,
    m: int = 5,
    rng_kind: str = "default_rng",
) -> np.ndarray:
    """Return a reproducible float64 ``(n, m)`` matrix, without global RNG use.

    Reported OOD experiments use ``default_rng``. The optional ``random_state``
    mode is restricted to the uniform control for legacy initialization checks;
    it is not the generator of the reported OOD data.
    """
    if distribution not in AVAILABLE_DISTRIBUTIONS:
        raise ValueError(f"unknown initial distribution: {distribution!r}")
    n, m, seed = operator.index(n), operator.index(m), operator.index(seed)
    if n <= 0 or m <= 0 or seed < 0:
        raise ValueError("n and m must be positive; seed must be nonnegative")
    if rng_kind == "default_rng":
        rng = np.random.default_rng(seed)
    elif rng_kind in ("random_state", "RandomState") and distribution == "uniform_control":
        rng = np.random.RandomState(seed)
    else:
        raise ValueError("OOD sampling requires default_rng; random_state is uniform-only")
    shape = (n, m)
    if distribution == "uniform_control":
        opinions = rng.uniform(0.0, 1.0, size=shape)
    elif distribution == "beta_center":
        opinions = rng.beta(2.0, 2.0, size=shape)
    elif distribution == "beta_extremes":
        opinions = rng.beta(0.5, 0.5, size=shape)
    elif distribution == "beta_high":
        opinions = rng.beta(5.0, 2.0, size=shape)
    else:
        # Preserve draw order: low rows, high rows, then permutation of experts.
        # Each expert stays in one group across all alternatives.
        opinions = np.empty(shape, dtype=np.float64)
        opinions[: n // 2] = rng.beta(2.0, 8.0, size=(n // 2, m))
        opinions[n // 2 :] = rng.beta(8.0, 2.0, size=(n - n // 2, m))
        opinions = opinions[rng.permutation(n)]
    if opinions.shape != shape or not np.all(np.isfinite(opinions)):
        raise RuntimeError("invalid generated initial opinion matrix")
    if np.any(opinions < 0.0) or np.any(opinions > 1.0):
        raise RuntimeError("generated opinions exceed [0,1]")
    return opinions
