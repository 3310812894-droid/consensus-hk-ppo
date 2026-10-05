"""Paper scenario grids and reproducible generators without policy execution."""

from dataclasses import asdict
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from consensus_hk.baselines import METHODS
from consensus_hk.evaluate import episode_jobs
from consensus_hk.io import summarize
from consensus_hk.model_factory import PPOConfig
from consensus_hk.protocol import (
    ABLATIONS, CONFIGURATIONS, DISTRIBUTIONS, EPSILONS, EVALUATION_SEEDS,
    OOD_EPSILONS, ORDINAL_WEIGHTS, SCALES, STABILITY_SEEDS, TRAINING_SEED,
    TRAINING_STEPS, TYPICAL_EPSILONS, configuration_parameters,
    ordinal_configuration, scale_configuration,
)
from consensus_hk.sampling import OOD_DISTRIBUTIONS, sample_initial_opinions
from tools import check_release


class ReleaseCheckTests(unittest.TestCase):
    def test_release_check_rejects_root_environment_files_without_name_error(self):
        root = Path("release-test-root")
        paths = [root / ".env", root / ".env.production"]
        archived_rows = [
            [{"model": model} for model in ("full_model", "no_mask", "no_gravity_penalty")],
            [{"model": model} for model in (
                "full_model", "hk", "scod", "adaptive_confidence", "noise_hk", "inertial")],
            [{"distribution": name} for name in (
                "beta_center", "beta_extremes", "beta_high", "bimodal")],
        ]
        output = StringIO()
        with (
            patch.object(check_release, "ROOT", root),
            patch.object(Path, "rglob", return_value=paths),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "read_text", return_value='{"files": []}') as read_text,
            patch.object(check_release, "EXPECTED_ROWS", {}),
            patch.object(check_release, "read_rows", side_effect=archived_rows),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as rejected,
        ):
            check_release.main()
        self.assertEqual(rejected.exception.code, 1)
        self.assertEqual(output.getvalue().splitlines(), [
            "Environment/secret file: .env",
            "Environment/secret file: .env.production",
        ])
        read_text.assert_called_once_with(encoding="utf-8")

    def test_release_file_selection_skips_environment_files_in_excluded_dirs(self):
        root = Path("release-test-root")
        included = root / "README.md"
        paths = [included]
        for directory in check_release.EXCLUDED_DIRS:
            paths.extend((root / directory / ".env", root / directory / ".env.production",
                          root / directory / "nested" / "artifact.txt"))
        with (
            patch.object(check_release, "ROOT", root),
            patch.object(Path, "rglob", return_value=paths),
            patch.object(Path, "is_file", return_value=True),
        ):
            self.assertEqual(list(check_release.included_files()), [included])


class ProtocolTests(unittest.TestCase):
    def test_exactly_two_ablations_five_baselines_and_four_ood_distributions(self):
        self.assertEqual(ABLATIONS, ("no_mask", "no_gravity_penalty"))
        self.assertEqual(len(METHODS), 5)
        self.assertEqual(set(METHODS), {"hk", "scod", "adaptive_confidence", "noise_hk", "inertial"})
        self.assertEqual(DISTRIBUTIONS, OOD_DISTRIBUTIONS)
        self.assertEqual(set(DISTRIBUTIONS), {"beta_center", "beta_extremes", "beta_high", "bimodal"})
        forbidden = {"no_ordinal_penalty", "no_both_penalties", "no_gravity_control"}
        self.assertFalse(forbidden.intersection(CONFIGURATIONS))
        for name in forbidden:
            with self.subTest(name=name), self.assertRaises(ValueError):
                configuration_parameters(name)

    def test_grid_episode_counts_and_unique_jobs(self):
        expected_counts = {
            "case": 3, "baselines": 60, "ordinal": 60,
            "scalability": 30, "ablation": 60, "sensitivity": 240,
            "stability": 100, "epsilon_ood": 40, "distribution_ood": 480,
        }
        for section, count in expected_counts.items():
            with self.subTest(section=section):
                jobs = episode_jobs(section)
                self.assertEqual(len(jobs), count)
                identifiers = [tuple(job[field] for field in (
                    "configuration", "epsilon", "seed", "N", "M", "distribution", "rng"))
                    for job in jobs]
                self.assertEqual(len(set(identifiers)), count)
                self.assertTrue(all(job["configuration"] in CONFIGURATIONS for job in jobs))

    def test_grid_sets_match_the_declared_paper_protocol(self):
        scales = episode_jobs("scalability")
        self.assertEqual({(j["N"], j["M"]) for j in scales}, set(SCALES))
        self.assertEqual({j["epsilon"] for j in scales}, set(EPSILONS))
        self.assertEqual({j["seed"] for j in scales}, {42})
        for job in scales:
            self.assertEqual(job["configuration"], scale_configuration(job["N"], job["M"]))
        ablations = episode_jobs("ablation")
        self.assertEqual({j["configuration"] for j in ablations}, {"main", *ABLATIONS})
        self.assertEqual({j["epsilon"] for j in ablations}, {0.05})
        self.assertEqual({j["seed"] for j in ablations}, set(EVALUATION_SEEDS))
        sensitivity = episode_jobs("sensitivity")
        self.assertEqual({j["lambda_rev"] for j in sensitivity}, set(ORDINAL_WEIGHTS))
        self.assertEqual({j["epsilon"] for j in sensitivity}, set(TYPICAL_EPSILONS))
        for job in sensitivity:
            self.assertEqual(job["configuration"], ordinal_configuration(job["lambda_rev"]))
        self.assertEqual({j["seed"] for j in episode_jobs("stability")}, set(STABILITY_SEEDS))
        self.assertEqual({j["epsilon"] for j in episode_jobs("epsilon_ood")}, set(OOD_EPSILONS))
        distribution = episode_jobs("distribution_ood")
        self.assertEqual({j["distribution"] for j in distribution}, set(DISTRIBUTIONS))
        self.assertEqual({j["epsilon"] for j in distribution}, set(EPSILONS))

    def test_typical_case_uses_randomstate_and_other_jobs_use_default_rng(self):
        case = episode_jobs("case")
        self.assertEqual({j["rng"] for j in case}, {"random_state"})
        self.assertEqual({j["seed"] for j in case}, {42})
        self.assertEqual({j["epsilon"] for j in case}, set(TYPICAL_EPSILONS))
        generated = sample_initial_opinions("uniform_control", 42, rng_kind=case[0]["rng"])
        np.testing.assert_array_equal(generated, np.random.RandomState(42).uniform(0, 1, (100, 5)))
        for section in ("baselines", "ordinal", "ablation", "sensitivity", "scalability",
                        "stability", "epsilon_ood", "distribution_ood"):
            self.assertEqual({j["rng"] for j in episode_jobs(section)}, {"default_rng"})

    def test_main_training_and_sensitive_coefficients(self):
        self.assertEqual((TRAINING_SEED, TRAINING_STEPS), (42, 10_000_000))
        self.assertEqual(PPOConfig().as_dict(), {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 1024,
            "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
        })
        for weight in ORDINAL_WEIGHTS:
            parameters = configuration_parameters(ordinal_configuration(weight))
            self.assertEqual(parameters["reward_config"].ordinal_weight, weight)
            self.assertEqual(parameters["reward_config"].gravity_threshold, 0.45)
            self.assertTrue(parameters["use_action_mask"])
        reward = asdict(configuration_parameters("main")["reward_config"])
        self.assertEqual(reward, {
            "variance_weight": 200.0, "gravity_weight": 0.05, "ordinal_weight": 1.0,
            "time_friction": 0.02, "terminal_bonus": 20.0, "gravity_threshold": 0.45,
        })

    def test_ood_sampling_repeats_and_does_not_consume_global_rng(self):
        before = np.random.get_state()
        for name in DISTRIBUTIONS:
            with self.subTest(distribution=name):
                first = sample_initial_opinions(name, 10000, n=7, m=3)
                second = sample_initial_opinions(name, 10000, n=7, m=3)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, (7, 3))
                self.assertEqual(first.dtype, np.float64)
                self.assertTrue(np.all((first >= 0) & (first <= 1)))
                with self.assertRaises(ValueError):
                    sample_initial_opinions(name, 42, rng_kind="random_state")
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_summary_success_population_and_sample_standard_deviation(self):
        shared = dict(model="main", N=100, M=5, epsilon=0.05,
                      distribution="uniform_control", lambda_rev=1.0)
        rows = [
            dict(shared, joint_success=True, intervention_steps=100,
                 equivalent_rounds=1.0, final_variance=0.02, gravity_shift=0.2, final_oprr=20.0),
            dict(shared, joint_success="True", intervention_steps=300,
                 equivalent_rounds=3.0, final_variance=0.04, gravity_shift=0.3, final_oprr=40.0),
            dict(shared, joint_success=False, intervention_steps=20000,
                 equivalent_rounds=200.0, final_variance=0.06, gravity_shift=0.5, final_oprr=60.0),
        ]
        summary, = summarize(rows)
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["joint_success_count"], 2)
        self.assertEqual(summary["intervention_steps_mean"], 200.0)
        self.assertAlmostEqual(summary["intervention_steps_std"], np.sqrt(20000.0))
        self.assertEqual(summary["equivalent_rounds_mean"], 2.0)
        self.assertEqual(summary["final_oprr_mean"], 40.0)
        self.assertEqual(summary["final_oprr_std"], 20.0)


if __name__ == "__main__":
    unittest.main()
