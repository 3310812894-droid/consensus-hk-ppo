"""Gymnasium environment for PPO-controlled multidimensional HK consensus."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_mask import apply_boundary_action_mask, scale_continuous_action
from .hk_dynamics import calculate_distance_matrix, hk_update
from .metrics import (calculate_final_oprr, calculate_gravity_shift,
                      calculate_total_variance, calculate_training_ordinal_cost)
from .reward import RewardConfig, calculate_reward


class ConsensusEnv(gym.Env):
    """One expert intervention per step, with synchronous HK after each sweep.

    Observations are ``[vec(U), vec(U0), vec(D), expert/N, V/(0.25*M), epsilon]``.
    Episodes terminate when both ``V <= variance_threshold`` and
    ``G <= gravity_threshold`` hold, or truncate after ``max_rounds`` sweeps.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(
        self,
        num_experts: int = 100,
        num_alternatives: int = 5,
        max_rounds: int = 200,
        micro_scale: float = 0.05,
        variance_threshold: float = 0.05,
        epsilon_min: float = 0.05,
        epsilon_max: float = 0.30,
        opinion_low: float = 0.0,
        opinion_high: float = 1.0,
        mask_low: float = 0.05,
        mask_high: float = 0.95,
        reward_config: Optional[RewardConfig] = None,
        use_action_mask: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0 or num_alternatives <= 0:
            raise ValueError("num_experts and num_alternatives must be positive")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        if not 0.0 <= epsilon_min <= epsilon_max <= 1.0:
            raise ValueError("epsilon range must lie in the physical range [0, 1]")
        self.N = int(num_experts)
        self.M = int(num_alternatives)
        self.max_rounds = int(max_rounds)
        self.micro_scale = float(micro_scale)
        self.variance_threshold = float(variance_threshold)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_max = float(epsilon_max)
        self.opinion_low = float(opinion_low)
        self.opinion_high = float(opinion_high)
        self.mask_low = float(mask_low)
        self.mask_high = float(mask_high)
        self.reward_config = reward_config or RewardConfig()
        self.use_action_mask = bool(use_action_mask)
        self.variance_max = 0.25 * self.M
        observation_size = 2 * self.N * self.M + self.N * self.N + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(observation_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.M,), dtype=np.float32)
        self.U_0: Optional[np.ndarray] = None
        self.U_t: Optional[np.ndarray] = None
        self.epsilon: Optional[float] = None
        self.current_expert = 0
        self.current_round = 0
        self.total_variance = 0.0
        self.gravity_shift = 0.0
        self.training_ordinal_cost = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        """Sample an episode, or accept explicit initial opinions and epsilon."""
        super().reset(seed=seed)
        options = options or {}
        fixed_initial_state = options.get("fixed_initial_state")
        if fixed_initial_state is None:
            self.U_0 = self.np_random.uniform(self.opinion_low, self.opinion_high,
                                             size=(self.N, self.M))
        else:
            initial = np.asarray(fixed_initial_state, dtype=np.float64)
            if initial.shape != (self.N, self.M):
                raise ValueError(f"fixed_initial_state must have shape {(self.N, self.M)}")
            if not np.all(np.isfinite(initial)):
                raise ValueError("fixed_initial_state must contain only finite values")
            self.U_0 = np.clip(initial.copy(), self.opinion_low, self.opinion_high)
        self.U_t = self.U_0.copy()
        fixed_epsilon = options.get("fixed_epsilon")
        if fixed_epsilon is None:
            self.epsilon = float(self.np_random.uniform(self.epsilon_min, self.epsilon_max))
        else:
            self.epsilon = float(fixed_epsilon)
            if not np.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
                raise ValueError("fixed_epsilon must lie in the physical range [0, 1]")
        self.current_expert = 0
        self.current_round = 0
        self.total_variance = calculate_total_variance(self.U_t)
        self.gravity_shift = 0.0
        self.training_ordinal_cost = 0.0
        info = {"initial_variance": self.total_variance, "epsilon": self.epsilon,
                "training_ordinal_cost": self.training_ordinal_cost}
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        """Apply action, optional boundary mask, sweep HK, metrics, then reward."""
        if self.U_t is None or self.U_0 is None or self.epsilon is None:
            raise RuntimeError("reset() must be called before step()")
        acted_expert = self.current_expert
        previous_variance = self.total_variance
        previous_gravity = self.gravity_shift
        previous_ordinal_cost = self.training_ordinal_cost
        physical_action = scale_continuous_action(action, self.micro_scale)
        if self.use_action_mask:
            masked_action = apply_boundary_action_mask(
                self.U_t[acted_expert], physical_action,
                lower_boundary=self.mask_low, upper_boundary=self.mask_high)
        else:
            masked_action = physical_action.copy()
        self.U_t[acted_expert] = np.clip(self.U_t[acted_expert] + masked_action,
                                       self.opinion_low, self.opinion_high)
        self.current_expert += 1
        hk_applied = self.current_expert >= self.N
        if hk_applied:
            self.current_expert = 0
            self.current_round += 1
            self.U_t = hk_update(self.U_t, self.epsilon, opinion_low=self.opinion_low,
                                 opinion_high=self.opinion_high)
        self.total_variance = calculate_total_variance(self.U_t)
        current_gravity = calculate_gravity_shift(self.U_0, self.U_t)
        self.gravity_shift = current_gravity
        self.training_ordinal_cost = calculate_training_ordinal_cost(self.U_0, self.U_t)
        incremental_ordinal_cost = self.training_ordinal_cost - previous_ordinal_cost
        variance_ok = self.total_variance <= self.variance_threshold
        gravity_ok = current_gravity <= self.reward_config.gravity_threshold
        terminated = variance_ok and gravity_ok
        truncated = self.current_round >= self.max_rounds and not terminated
        components = calculate_reward(
            previous_variance=previous_variance, current_variance=self.total_variance,
            previous_gravity=previous_gravity, current_gravity=current_gravity,
            previous_ordinal_cost=previous_ordinal_cost,
            current_ordinal_cost=self.training_ordinal_cost,
            terminated=terminated, config=self.reward_config)
        final_oprr = calculate_final_oprr(self.U_0, self.U_t) if terminated or truncated else None
        info = {
            "variance": self.total_variance,
            "gravity_shift": current_gravity,
            "previous_gravity": previous_gravity,
            "current_gravity": current_gravity,
            "delta_gravity": current_gravity - previous_gravity,
            "training_ordinal_cost": self.training_ordinal_cost,
            "previous_ordinal_cost": previous_ordinal_cost,
            "current_ordinal_cost": self.training_ordinal_cost,
            "delta_ordinal_cost": incremental_ordinal_cost,
            "incremental_ordinal_cost": incremental_ordinal_cost,
            "final_oprr": final_oprr,
            "round": self.current_round,
            "current_expert": self.current_expert,
            "acted_expert": acted_expert,
            "epsilon": self.epsilon,
            "hk_applied": hk_applied,
            "physical_action": physical_action.copy(),
            "masked_action": masked_action.copy(),
            "variance_feasible": variance_ok,
            "gravity_feasible": gravity_ok,
        }
        info.update(components.as_dict())
        return self._get_obs(), components.total, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self.U_t is None or self.U_0 is None or self.epsilon is None:
            raise RuntimeError("environment state is not initialized")
        distance_matrix = calculate_distance_matrix(self.U_t)
        observation = np.concatenate([
            self.U_t.ravel(), self.U_0.ravel(), distance_matrix.ravel(),
            np.asarray([self.current_expert / self.N], dtype=np.float64),
            np.asarray([self.total_variance / self.variance_max], dtype=np.float64),
            np.asarray([self.epsilon], dtype=np.float64),
        ])
        return observation.astype(np.float32)

    def render(self) -> None:
        print(f"round={self.current_round}, expert={self.current_expert}, "
              f"variance={self.total_variance:.6f}, epsilon={self.epsilon:.4f}")
