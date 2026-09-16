"""
Agent run configurations and scripts for trained Dino RL agents.
"""
from .agents_config import DINO_DQN_V1, DINO_DQN_V2, DINOV1
from .dinoV1 import run_dinoV1

__all__ = [
    "DINO_DQN_V1",
    "DINO_DQN_V2",
    "DINOV1",
    "run_dinoV1",
]
