"""Five-component Reward-v4 with positive-increment shaping penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Coefficients and gravity threshold of the paper-scale main model."""

    variance_weight: float = 200.0
    gravity_weight: float = 0.05
    ordinal_weight: float = 1.0
    time_friction: float = 0.02
    terminal_bonus: float = 20.0
    gravity_threshold: float = 0.45


@dataclass(frozen=True)
class RewardComponents:
    """Reward terms and the endpoints used to calculate shaping penalties."""

    variance_reward: float
    gravity_penalty: float
    ordinal_penalty: float
    time_penalty: float
    terminal_reward: float
    previous_gravity: float
    current_gravity: float
    delta_gravity: float
    positive_delta_gravity: float
    previous_ordinal_cost: float
    current_ordinal_cost: float
    delta_ordinal_cost: float

    @property
    def total(self) -> float:
        return float(self.variance_reward + self.gravity_penalty + self.ordinal_penalty
                     + self.time_penalty + self.terminal_reward)

    def as_dict(self) -> dict[str, float]:
        return {
            "reward_variance": self.variance_reward,
            "reward_gravity": self.gravity_penalty,
            "reward_ordinal": self.ordinal_penalty,
            "reward_time": self.time_penalty,
            "reward_terminal": self.terminal_reward,
            "reward_total": self.total,
            "previous_gravity": self.previous_gravity,
            "current_gravity": self.current_gravity,
            "delta_gravity": self.delta_gravity,
            "positive_delta_gravity": self.positive_delta_gravity,
            "previous_ordinal_cost": self.previous_ordinal_cost,
            "current_ordinal_cost": self.current_ordinal_cost,
            "delta_ordinal_cost": self.delta_ordinal_cost,
        }


def calculate_reward(
    *,
    previous_variance: float,
    current_variance: float,
    previous_gravity: float,
    current_gravity: float,
    previous_ordinal_cost: float,
    current_ordinal_cost: float,
    terminated: bool,
    config: RewardConfig,
) -> RewardComponents:
    """Calculate reward from the complete successor state.

    Gravity and ordinal penalties use ``max(0, current - previous)``.
    Reductions in either cost do not reimburse previous penalties.
    """
    values = np.asarray([previous_variance, current_variance, previous_gravity,
                         current_gravity, previous_ordinal_cost, current_ordinal_cost],
                        dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("reward inputs must be finite")
    delta_gravity = float(current_gravity) - float(previous_gravity)
    positive_delta_gravity = max(0.0, delta_gravity)
    delta_ordinal_cost = float(current_ordinal_cost) - float(previous_ordinal_cost)
    return RewardComponents(
        variance_reward=config.variance_weight * (previous_variance - current_variance),
        gravity_penalty=-config.gravity_weight * positive_delta_gravity,
        ordinal_penalty=-config.ordinal_weight * max(0.0, delta_ordinal_cost),
        time_penalty=-config.time_friction,
        terminal_reward=config.terminal_bonus if terminated else 0.0,
        previous_gravity=float(previous_gravity),
        current_gravity=float(current_gravity),
        delta_gravity=delta_gravity,
        positive_delta_gravity=positive_delta_gravity,
        previous_ordinal_cost=float(previous_ordinal_cost),
        current_ordinal_cost=float(current_ordinal_cost),
        delta_ordinal_cost=delta_ordinal_cost,
    )
