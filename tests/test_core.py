"""Environment and metric contracts without checkpoint loading or learning."""

from dataclasses import asdict, replace
import unittest

import numpy as np

from consensus_hk.environment import ConsensusEnv
from consensus_hk.hk_dynamics import calculate_distance_matrix, hk_update
from consensus_hk.metrics import (
    calculate_final_oprr, calculate_training_ordinal_cost, strict_reversal_counts,
)
from consensus_hk.protocol import configuration_parameters, make_env
from consensus_hk.reward import RewardConfig, calculate_reward
from consensus_hk.sampling import sample_initial_opinions


class EnvironmentTests(unittest.TestCase):
    def test_variance_alone_never_terminates_either_gravity_configuration(self):
        initial = np.array([[0.1, 0.2]])
        for gravity_weight in (0.05, 0.0):
            with self.subTest(gravity_weight=gravity_weight):
                env = ConsensusEnv(
                    num_experts=1, num_alternatives=2, max_rounds=2,
                    micro_scale=0.6,
                    reward_config=replace(RewardConfig(), gravity_weight=gravity_weight),
                )
                self.addCleanup(env.close)
                env.reset(seed=42, options={"fixed_initial_state": initial, "fixed_epsilon": 0.05})
                _, _, terminated, truncated, info = env.step(np.ones(2))
                self.assertTrue(info["variance_feasible"])
                self.assertFalse(info["gravity_feasible"])
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertAlmostEqual(info["current_gravity"], 0.6)
                self.assertEqual(info["reward_terminal"], 0.0)
                _, _, terminated, truncated, _ = env.step(np.zeros(2))
                self.assertFalse(terminated)
                self.assertTrue(truncated)

    def test_joint_success_terminates_and_receives_bonus(self):
        env = ConsensusEnv(num_experts=1, num_alternatives=2)
        self.addCleanup(env.close)
        env.reset(seed=42, options={"fixed_initial_state": np.array([[0.2, 0.8]])})
        _, reward, terminated, truncated, info = env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["variance_feasible"] and info["gravity_feasible"])
        self.assertEqual(info["reward_terminal"], 20.0)
        self.assertAlmostEqual(reward, 19.98)

    def test_no_mask_changes_only_boundary_action_path(self):
        main_parameters = configuration_parameters("main")
        unmasked_parameters = configuration_parameters("no_mask")
        self.assertEqual(
            {key for key in main_parameters if main_parameters[key] != unmasked_parameters[key]},
            {"use_action_mask"},
        )
        initial = np.array([[0.96, 0.04], [0.2, 0.8]])
        main = ConsensusEnv(num_experts=2, num_alternatives=2)
        unmasked = ConsensusEnv(num_experts=2, num_alternatives=2, use_action_mask=False)
        self.addCleanup(main.close)
        self.addCleanup(unmasked.close)
        for env in (main, unmasked):
            env.reset(seed=42, options={"fixed_initial_state": initial, "fixed_epsilon": 0.05})
        _, _, _, _, main_info = main.step(np.array([1.0, -1.0]))
        _, _, _, _, unmasked_info = unmasked.step(np.array([1.0, -1.0]))
        np.testing.assert_array_equal(main_info["physical_action"], unmasked_info["physical_action"])
        np.testing.assert_array_equal(main_info["masked_action"], np.zeros(2))
        np.testing.assert_array_equal(unmasked_info["masked_action"], np.array([0.05, -0.05]))
        np.testing.assert_array_equal(main.U_t, initial)
        np.testing.assert_array_equal(unmasked.U_t, np.array([[1.0, 0.0], [0.2, 0.8]]))
        self.assertEqual(main.current_expert, unmasked.current_expert)
        self.assertEqual(main.current_round, unmasked.current_round)

    def test_no_gravity_penalty_changes_only_weight_and_reward(self):
        main_parameters = configuration_parameters("main")
        ablated_parameters = configuration_parameters("no_gravity_penalty")
        main_reward = asdict(main_parameters.pop("reward_config"))
        ablated_reward = asdict(ablated_parameters.pop("reward_config"))
        self.assertEqual(main_parameters, ablated_parameters)
        self.assertEqual(
            {key for key in main_reward if main_reward[key] != ablated_reward[key]},
            {"gravity_weight"},
        )
        self.assertEqual(ablated_reward["gravity_weight"], 0.0)
        self.assertEqual(ablated_reward["gravity_threshold"], 0.45)
        initial = np.array([[0.2, 0.8], [0.4, 0.6], [0.8, 0.2]])
        main = ConsensusEnv(num_experts=3, num_alternatives=2)
        ablated = ConsensusEnv(
            num_experts=3, num_alternatives=2,
            reward_config=replace(RewardConfig(), gravity_weight=0.0),
        )
        self.addCleanup(main.close)
        self.addCleanup(ablated.close)
        for env in (main, ablated):
            env.reset(seed=42, options={"fixed_initial_state": initial, "fixed_epsilon": 0.05})
        obs, reward, terminated, truncated, info = main.step(np.array([1.0, -1.0]))
        other_obs, other_reward, other_terminated, other_truncated, other_info = ablated.step(
            np.array([1.0, -1.0]))
        np.testing.assert_array_equal(obs, other_obs)
        np.testing.assert_array_equal(main.U_t, ablated.U_t)
        self.assertEqual((terminated, truncated), (other_terminated, other_truncated))
        self.assertEqual(other_info["reward_gravity"], 0.0)
        for field in ("reward_variance", "reward_ordinal", "reward_time", "reward_terminal"):
            self.assertEqual(info[field], other_info[field])
        self.assertAlmostEqual(other_reward - reward, -info["reward_gravity"], places=14)

    def test_observation_layout_and_sweep_end_synchronous_hk(self):
        initial = np.array([[0.0, 0.0], [0.4, 0.4], [1.0, 1.0]])
        env = ConsensusEnv(num_experts=3, num_alternatives=2)
        self.addCleanup(env.close)
        observation, _ = env.reset(seed=42, options={
            "fixed_initial_state": initial, "fixed_epsilon": 0.7,
        })
        self.assertEqual(observation.shape, (24,))
        self.assertEqual(observation.dtype, np.float32)
        expected = np.concatenate([
            initial.ravel(), initial.ravel(), calculate_distance_matrix(initial).ravel(),
            [0.0, env.total_variance / 0.5, 0.7],
        ]).astype(np.float32)
        np.testing.assert_array_equal(observation, expected)
        for expected_expert in (1, 2):
            _, _, _, _, info = env.step(np.zeros(2))
            self.assertFalse(info["hk_applied"])
            self.assertEqual(env.current_expert, expected_expert)
            self.assertEqual(env.current_round, 0)
            np.testing.assert_array_equal(env.U_t, initial)
        _, _, _, _, info = env.step(np.zeros(2))
        self.assertTrue(info["hk_applied"])
        self.assertEqual((env.current_round, env.current_expert), (1, 0))
        expected_hk = np.array([[0.2, 0.2], [1.4 / 3, 1.4 / 3], [0.7, 0.7]])
        np.testing.assert_allclose(env.U_t, expected_hk, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(env.U_t, hk_update(initial, 0.7))

    def test_episode_generator_draws_u0_then_epsilon_and_keeps_stream(self):
        env = make_env("main")
        self.addCleanup(env.close)
        rng = np.random.default_rng(42)
        for seed in (42, None):
            expected_initial = rng.uniform(0.0, 1.0, size=(100, 5))
            expected_epsilon = float(rng.uniform(0.05, 0.30))
            _, info = env.reset(seed=seed)
            np.testing.assert_array_equal(env.U_0, expected_initial)
            np.testing.assert_array_equal(env.U_t, expected_initial)
            self.assertEqual(info["epsilon"], expected_epsilon)
        env.reset(seed=42)
        np.testing.assert_array_equal(env.U_0, sample_initial_opinions("uniform_control", 42))


class MetricAndRewardTests(unittest.TestCase):
    def test_strict_reversals_exclude_initial_and_terminal_ties(self):
        initial = np.array([[0.9, 0.4, 0.1], [0.5, 0.5, 0.2], [0.2, 0.2, 0.2]])
        current = np.array([[0.1, 0.4, 0.4], [0.1, 0.9, 0.5], [0.8, 0.4, 0.1]])
        valid, reversals, tied = strict_reversal_counts(initial, current)
        np.testing.assert_array_equal(valid, [3, 2, 0])
        np.testing.assert_array_equal(reversals, [2, 1, 0])
        np.testing.assert_array_equal(tied, [1, 0, 0])
        self.assertEqual(calculate_final_oprr(initial, current), 60.0)
        self.assertEqual(calculate_training_ordinal_cost(initial, current), 1.0)

    def test_no_initially_strict_pairs_produces_zero_oprr(self):
        initial = np.full((2, 3), 0.5)
        current = np.array([[0.1, 0.8, 0.2], [0.9, 0.4, 0.1]])
        for counts in strict_reversal_counts(initial, current):
            np.testing.assert_array_equal(counts, [0, 0])
        self.assertEqual(calculate_final_oprr(initial, current), 0.0)

    def test_positive_increment_penalties_and_terminal_bonus(self):
        components = calculate_reward(
            previous_variance=0.5, current_variance=0.4,
            previous_gravity=0.3, current_gravity=0.25,
            previous_ordinal_cost=3.0, current_ordinal_cost=2.0,
            terminated=False, config=RewardConfig(),
        )
        self.assertAlmostEqual(components.variance_reward, 20.0)
        self.assertEqual(components.gravity_penalty, 0.0)
        self.assertEqual(components.ordinal_penalty, 0.0)
        self.assertEqual(components.terminal_reward, 0.0)
        self.assertAlmostEqual(components.total, 19.98)
        components = calculate_reward(
            previous_variance=0.4, current_variance=0.4,
            previous_gravity=0.2, current_gravity=0.3,
            previous_ordinal_cost=1.0, current_ordinal_cost=2.0,
            terminated=True, config=RewardConfig(),
        )
        self.assertAlmostEqual(components.gravity_penalty, -0.005)
        self.assertEqual(components.ordinal_penalty, -1.0)
        self.assertEqual(components.time_penalty, -0.02)
        self.assertEqual(components.terminal_reward, 20.0)
        self.assertAlmostEqual(components.total, 18.975)

    def test_ordinal_penalty_remembers_positive_path_changes(self):
        cumulative = 0.0
        for previous, current in ((0.0, 2.0), (2.0, 1.0), (1.0, 2.0)):
            components = calculate_reward(
                previous_variance=0.4, current_variance=0.4,
                previous_gravity=0.0, current_gravity=0.0,
                previous_ordinal_cost=previous, current_ordinal_cost=current,
                terminated=False, config=RewardConfig(),
            )
            cumulative += components.ordinal_penalty
        self.assertEqual(cumulative, -3.0)


if __name__ == "__main__":
    unittest.main()
