"""Small-matrix checks of the published baseline rules, with no RL execution."""

from __future__ import annotations

import unittest

import numpy as np

from consensus_hk.baselines import METHODS, run_baseline


def explicit_trajectory(method, initial, epsilon, seed, rounds):
    """Independent, rowwise calculation of the five prespecified rules."""
    current = np.asarray(initial, dtype=np.float64).copy()
    n = current.shape[0]
    confidence = np.full((n, n), epsilon, dtype=np.float64)
    np.fill_diagonal(confidence, 0.0)
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E484B]))
    states = [current.copy()]
    for _ in range(rounds):
        old = current.copy()
        differences = np.abs(old[:, None, :] - old[None, :, :])
        distances = np.mean(differences, axis=2)
        if method == "scod":
            trusted = np.any(differences <= epsilon, axis=2)
        elif method == "adaptive_confidence":
            trusted = distances < confidence
        else:
            trusted = distances <= epsilon
        # Each successor uses only the same OLD opinion matrix.
        successor = np.empty_like(old)
        for expert in range(n):
            neighbors = trusted[expert].copy()
            neighbors[expert] = True
            successor[expert] = np.mean(old[neighbors], axis=0)
        if method == "adaptive_confidence":
            confidence = np.where(trusted, confidence + 0.1 * (1.0 - confidence),
                                  0.9 * confidence)
            np.fill_diagonal(confidence, 0.0)
        elif method == "inertial":
            successor = 0.5 * old + 0.5 * successor
        elif method == "noise_hk":
            successor += noise_rng.uniform(-epsilon / 4.0, epsilon / 4.0,
                                           size=old.shape)
        current = np.clip(successor, 0.0, 1.0)
        states.append(current.copy())
    return states


class BaselineTests(unittest.TestCase):
    def run_fixed(self, method, initial, epsilon, rounds=1, seed=10007):
        return run_baseline(method, initial, epsilon, seed, max_rounds=rounds,
                            stop_on_success=False, record_trajectory=True)

    def test_all_five_one_and_ten_steps_against_explicit_rules(self):
        self.assertEqual(set(METHODS),
                         {"hk", "scod", "adaptive_confidence", "noise_hk", "inertial"})
        initial = np.array([[0.10, 0.20, 0.30], [0.15, 0.25, 0.35],
                            [0.40, 0.35, 0.45], [0.90, 0.85, 0.80]])
        for method in METHODS:
            for rounds in (1, 10):
                with self.subTest(method=method, rounds=rounds):
                    row, trajectory = self.run_fixed(method, initial, 0.22, rounds)
                    expected = explicit_trajectory(method, initial, 0.22, 10007, rounds)
                    self.assertEqual(len(trajectory), rounds + 1)
                    self.assertEqual(row["completed_synchronous_updates"], rounds)
                    self.assertEqual(row["individual_intervention_steps"], 0)
                    for actual, matrix in zip(trajectory, expected):
                        np.testing.assert_allclose(actual["opinions"], matrix,
                                                   rtol=0.0, atol=5e-15)
                    final = trajectory[-1]["opinions"]
                    self.assertAlmostEqual(row["final_variance"],
                                           np.var(final, axis=0).sum(), places=14)
                    gravity = np.linalg.norm(final.mean(axis=0) - initial.mean(axis=0)) / np.sqrt(3)
                    self.assertAlmostEqual(row["gravity_shift"], gravity, places=14)

    def test_hk_includes_exact_confidence_boundary(self):
        initial = np.array([[0.0, 0.5], [0.25, 0.75], [1.0, 0.0]])
        _, trajectory = self.run_fixed("hk", initial, 0.25)
        expected = np.array([[0.125, 0.625], [0.125, 0.625], [1.0, 0.0]])
        np.testing.assert_array_equal(trajectory[1]["opinions"], expected)

    def test_scod_union_neighbors_average_whole_vectors(self):
        initial = np.array([[0.10, 0.90], [0.15, 0.20], [0.80, 0.25]])
        _, trajectory = self.run_fixed("scod", initial, 0.06)
        expected = np.array([[0.125, 0.55], [0.35, 0.45], [0.475, 0.225]])
        np.testing.assert_allclose(trajectory[1]["opinions"], expected,
                                   rtol=0.0, atol=1e-15)
        # The first expert averages the distant second coordinate too.
        self.assertNotEqual(trajectory[1]["opinions"][0, 1], initial[0, 1])

    def test_adaptive_excludes_exact_boundary_but_includes_self(self):
        initial = np.array([[0.0], [0.25]])
        _, trajectory = self.run_fixed("adaptive_confidence", initial, 0.25, 10)
        for state in trajectory:
            np.testing.assert_array_equal(state["opinions"], initial)

    def test_adaptive_confidence_updates_use_old_opinions(self):
        initial = np.array([[0.0], [0.24], [0.48]])
        _, trajectory = self.run_fixed("adaptive_confidence", initial, 0.25, 2)
        np.testing.assert_allclose(trajectory[1]["opinions"], [[0.12], [0.24], [0.36]],
                                   rtol=0.0, atol=1e-15)
        # Old outer distance .48 was unreceptive, so C_02 becomes .225.
        # New outer distance .24 is still excluded in step two. Updating C
        # from successor opinions instead would connect everyone too soon.
        np.testing.assert_allclose(trajectory[2]["opinions"], [[0.18], [0.24], [0.30]],
                                   rtol=0.0, atol=1e-15)

    def test_inertial_retention_is_one_half(self):
        initial = np.array([[0.0, 0.5], [0.25, 0.75], [1.0, 0.0]])
        _, trajectory = self.run_fixed("inertial", initial, 0.25)
        expected = np.array([[0.0625, 0.5625], [0.1875, 0.6875], [1.0, 0.0]])
        np.testing.assert_array_equal(trajectory[1]["opinions"], expected)

    def test_noise_stream_independence_order_and_clipping(self):
        initial = np.zeros((2, 2))
        saved_global_state = np.random.get_state()
        try:
            np.random.seed(12345)
            before = np.random.get_state()
            _, trajectory = self.run_fixed("noise_hk", initial, 1.0, seed=7)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            rng = np.random.default_rng(np.random.SeedSequence([7, 0x4E484B]))
            expected = np.clip(rng.uniform(-0.25, 0.25, size=(2, 2)), 0.0, 1.0)
            np.testing.assert_array_equal(trajectory[1]["opinions"], expected)
            # Unrelated initial-state and global RNG draws must not change noise.
            np.random.default_rng(7).uniform(size=(30, 20))
            np.random.uniform(size=100)
            _, repeat = self.run_fixed("noise_hk", initial, 1.0, seed=7)
            np.testing.assert_array_equal(repeat[1]["opinions"], expected)
            _, different = self.run_fixed("noise_hk", initial, 1.0, seed=8)
            self.assertFalse(np.array_equal(different[1]["opinions"], expected))
        finally:
            np.random.set_state(saved_global_state)

    def test_trajectory_arrays_are_independent_copies(self):
        initial = np.array([[0.0, 0.5], [0.25, 0.75], [1.0, 0.0]])
        original = initial.copy()
        _, trajectory = self.run_fixed("inertial", initial, 0.25, 3)
        np.testing.assert_array_equal(initial, original)
        for left in range(len(trajectory)):
            self.assertFalse(np.shares_memory(initial, trajectory[left]["opinions"]))
            for right in range(left + 1, len(trajectory)):
                self.assertFalse(np.shares_memory(trajectory[left]["opinions"],
                                                 trajectory[right]["opinions"]))
        last = trajectory[-1]["opinions"].copy()
        trajectory[0]["opinions"][:] = 0.0
        np.testing.assert_array_equal(initial, original)
        np.testing.assert_array_equal(trajectory[-1]["opinions"], last)
        _, absent = run_baseline("hk", initial, 0.25, 7, max_rounds=1)
        self.assertIsNone(absent)

    def test_zero_round_joint_success_and_terminal_not_ever_success(self):
        initial = np.zeros((2, 2))
        for method in METHODS:
            with self.subTest(method=method):
                row, trajectory = run_baseline(method, initial, 1.0, 7, max_rounds=3,
                                                record_trajectory=True)
                self.assertTrue(row["joint_success"])
                self.assertEqual(row["completed_synchronous_updates"], 0)
                self.assertEqual(len(trajectory), 1)
        row, _ = run_baseline("noise_hk", initial, 1.0, 7, max_rounds=1,
                              variance_threshold=1e-12, stop_on_success=False)
        self.assertEqual(row["first_joint_success_round"], 0.0)
        self.assertFalse(row["joint_success"])
        self.assertEqual(row["completed_synchronous_updates"], 1)
        self.assertIsNone(row["successful_equivalent_rounds"])

    def test_variance_only_hit_does_not_stop(self):
        initial = np.array([[0.0], [0.20], [0.45]])
        row, trajectory = run_baseline("hk", initial, 0.30, 7, max_rounds=5,
                                       variance_threshold=0.02, gravity_threshold=1e-10,
                                       record_trajectory=True)
        self.assertEqual(row["first_variance_round"], 1.0)
        self.assertTrue(trajectory[1]["variance_feasible"])
        self.assertFalse(trajectory[1]["gravity_feasible"])
        self.assertFalse(row["joint_success"])
        self.assertEqual(row["completed_synchronous_updates"], 5)
        self.assertEqual(row["stop_reason"], "max_rounds")


if __name__ == "__main__":
    unittest.main()
