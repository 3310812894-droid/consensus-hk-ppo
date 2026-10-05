"""PPO control of multidimensional Hegselmann--Krause opinion dynamics."""

from .environment import ConsensusEnv
from .reward import RewardConfig

__all__ = ["ConsensusEnv", "RewardConfig"]
