"""Training configurations and evaluation grids used in the paper."""

from __future__ import annotations

from dataclasses import replace

from .environment import ConsensusEnv
from .reward import RewardConfig


TRAINING_SEED = 42
TRAINING_STEPS = 10_000_000
EPSILONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TYPICAL_EPSILONS = (0.05, 0.15, 0.25)
OOD_EPSILONS = (0.03, 0.35)
EVALUATION_SEEDS = tuple(range(10_000, 10_020))
STABILITY_SEEDS = tuple(range(10_000, 10_100))
SCALES = ((50, 5), (100, 5), (200, 5), (100, 3), (100, 10))
ORDINAL_WEIGHTS = (0.5, 1.0, 1.5, 2.0)
DISTRIBUTIONS = ("beta_center", "beta_extremes", "beta_high", "bimodal")
ABLATIONS = ("no_mask", "no_gravity_penalty")
CONFIGURATIONS = (
    "main", *ABLATIONS, "lambda_rev_0p5", "lambda_rev_1p5", "lambda_rev_2",
    "N_50_M_5", "N_200_M_5", "N_100_M_3", "N_100_M_10",
)


def ordinal_configuration(weight: float) -> str:
    return {
        0.5: "lambda_rev_0p5", 1.0: "main",
        1.5: "lambda_rev_1p5", 2.0: "lambda_rev_2",
    }[weight]


def scale_configuration(n: int, m: int) -> str:
    return "main" if (n, m) == (100, 5) else f"N_{n}_M_{m}"


def configuration_parameters(name: str) -> dict:
    if name not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {name}")
    n, m = 100, 5
    if name.startswith("N_"):
        _, n_text, _, m_text = name.split("_")
        n, m = int(n_text), int(m_text)
    weight = {"lambda_rev_0p5": 0.5, "lambda_rev_1p5": 1.5,
              "lambda_rev_2": 2.0}.get(name, 1.0)
    return {
        "num_experts": n, "num_alternatives": m, "max_rounds": 200,
        "micro_scale": 0.05, "variance_threshold": 0.05,
        "epsilon_min": 0.05, "epsilon_max": 0.30,
        "use_action_mask": name != "no_mask",
        "reward_config": replace(
            RewardConfig(), ordinal_weight=weight,
            gravity_weight=0.0 if name == "no_gravity_penalty" else 0.05,
        ),
    }


def make_env(name: str = "main") -> ConsensusEnv:
    return ConsensusEnv(**configuration_parameters(name))

