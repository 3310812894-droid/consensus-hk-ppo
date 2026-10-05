"""Train one paper configuration; model files remain local and git-ignored."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

import csv

from .io import create_output_directory, runtime_versions, write_json
from .model_factory import PPOConfig, create_ppo_model, make_vector_env
from .protocol import (CONFIGURATIONS, TRAINING_SEED, TRAINING_STEPS,
                       configuration_parameters, make_env)


class TrainingMetricsCallback(BaseCallback):
    """Append rollout diagnostics without affecting policy updates."""

    def __init__(self, path: Path):
        super().__init__(verbose=0)
        self.path = path
        self.fields = ("record_type", "timestep", "learning_rate", "explained_variance",
                       "policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl",
                       "clip_fraction", "episode_reward_mean", "episode_length_mean")

    def _on_training_start(self):
        with self.path.open("x", encoding="utf-8-sig", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def _on_step(self):
        return True

    def _record(self, record_type):
        values = self.logger.name_to_value
        row = dict(record_type=record_type, timestep=int(self.num_timesteps))
        for name in self.fields[2:]:
            logger_key = {"episode_reward_mean": "rollout/ep_rew_mean",
                          "episode_length_mean": "rollout/ep_len_mean"}.get(name, f"train/{name}")
            row[name] = values.get(logger_key, "")
        with self.path.open("a", encoding="utf-8-sig", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)

    def _on_rollout_end(self):
        self._record("rollout_end")

    def _on_training_end(self):
        self._record("training_end")


def train(configuration: str, output: Path, timesteps: int, device: str) -> Path:
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    output = create_output_directory(output)
    parameters = configuration_parameters(configuration)
    reward = parameters.pop("reward_config")
    write_json(output / "configuration.json", {
        "configuration": configuration, "seed": TRAINING_SEED,
        "requested_timesteps": timesteps, "paper_budget": timesteps == TRAINING_STEPS,
        "environment": parameters, "reward": asdict(reward),
        "ppo": PPOConfig().as_dict(), "n_envs": 1,
        "network": "SB3 MlpPolicy default: separate [64,64] actor and critic",
        "training_initial_rng": "Gymnasium np_random Generator; U0 then epsilon",
        "runtime": runtime_versions(), "requested_device": device,
    })
    vec_env = make_vector_env(
        lambda: make_env(configuration), n_envs=1, seed=TRAINING_SEED,
        monitor_dir=output / "monitor",
    )
    model = create_ppo_model(
        vec_env, config=PPOConfig(), seed=TRAINING_SEED,
        tensorboard_log=output / "tensorboard", device=device, verbose=1,
    )
    callback = CallbackList([
        TrainingMetricsCallback(output / "training_metrics.csv"),
        CheckpointCallback(
            save_freq=500_000, save_path=str(output / "checkpoints"),
            name_prefix="ppo", save_replay_buffer=False, save_vecnormalize=False,
        ),
    ])
    status = "failed"
    try:
        model.learn(total_timesteps=timesteps, callback=callback)
        checkpoint = output / "final_model.zip"
        model.save(str(checkpoint))
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        checkpoint = output / "interrupted_model.zip"
        model.save(str(checkpoint))
        raise
    finally:
        try:
            write_json(output / "training_status.json", {
                "status": status, "requested_timesteps": timesteps,
                "actual_timesteps": int(model.num_timesteps),
                "device": str(model.device),
            })
        finally:
            vec_env.close()
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="main")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timesteps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output = args.output or Path("runs") / args.configuration
    print(train(args.configuration, output, args.timesteps, args.device))


if __name__ == "__main__":
    main()
