"""Stable-Baselines3 PPO construction and checkpoint compatibility helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters passed directly to Stable-Baselines3."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Return a learning rate proportional to remaining training progress."""
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * float(initial_value)
    return schedule


def make_vector_env(
    env_factory: Callable[[], gym.Env],
    n_envs: int = 1,
    seed: Optional[int] = None,
    monitor_dir: Optional[Union[str, Path]] = None,
    use_subprocesses: bool = False,
) -> VecEnv:
    """Create a monitored vector environment from a Gymnasium factory."""
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    monitor_path = None
    if monitor_dir is not None:
        monitor_path = Path(monitor_dir)
        monitor_path.mkdir(parents=True, exist_ok=True)
    vec_env_class = SubprocVecEnv if use_subprocesses and n_envs > 1 else DummyVecEnv
    return make_vec_env(env_factory, n_envs=n_envs, seed=seed,
                        monitor_dir=str(monitor_path) if monitor_path is not None else None,
                        vec_env_cls=vec_env_class)


def create_ppo_model(
    env: VecEnv,
    config: Optional[PPOConfig] = None,
    seed: Optional[int] = None,
    tensorboard_log: Optional[Union[str, Path]] = None,
    device: str = "auto",
    verbose: int = 1,
) -> PPO:
    """Create PPO with the standard MlpPolicy and its default architecture."""
    parameters = config or PPOConfig()
    tensorboard_path = str(tensorboard_log) if tensorboard_log is not None else None
    return PPO(
        "MlpPolicy", env,
        learning_rate=linear_schedule(parameters.learning_rate),
        n_steps=parameters.n_steps, batch_size=parameters.batch_size,
        n_epochs=parameters.n_epochs, gamma=parameters.gamma,
        gae_lambda=parameters.gae_lambda, clip_range=parameters.clip_range,
        ent_coef=parameters.ent_coef, vf_coef=parameters.vf_coef,
        max_grad_norm=parameters.max_grad_norm, seed=seed,
        tensorboard_log=tensorboard_path, device=device, verbose=verbose,
    )


def assert_model_env_compatible(model: PPO, env: VecEnv) -> None:
    """Reject mismatched observation or action dimensions."""
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(f"checkpoint observation shape {model.observation_space.shape} "
                         f"does not match environment {env.observation_space.shape}")
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(f"checkpoint action shape {model.action_space.shape} "
                         f"does not match environment {env.action_space.shape}")


def load_ppo_checkpoint(
    checkpoint_path: Union[str, Path],
    env: Optional[VecEnv] = None,
    device: str = "auto",
) -> PPO:
    """Load a PPO checkpoint and optionally attach a compatible environment."""
    path = Path(checkpoint_path)
    if not path.exists() and path.suffix != ".zip":
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")
    model = PPO.load(str(path), env=env, device=device)
    if env is not None:
        assert_model_env_compatible(model, env)
    return model


def save_ppo_checkpoint(model: PPO, checkpoint_path: Union[str, Path]) -> Path:
    """Save a PPO checkpoint and return the resulting ZIP path."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path if path.suffix == ".zip" else path.with_suffix(".zip")
