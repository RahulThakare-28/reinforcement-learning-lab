"""
Agents package for Chrome Dino reinforcement learning architectures.
"""
from . import dinoV1
from .agents_config import DINO_DQN_V1, DINO_DQN_V2, DINOV1

__all__ = [
    "dinoV1",
    "DINO_DQN_V1",
    "DINO_DQN_V2",
    "DINOV1",
]