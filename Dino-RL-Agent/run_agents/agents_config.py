"""
Model path configuration for trained Dino RL agents.
Re-exported for run_agents package.
"""

import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_here, ".."))
SRC_PATH = os.path.abspath(os.path.join(_here, "..", "src"))

for p in [PROJECT_ROOT, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.agents.agents_config import DINO_DQN_V1, DINO_DQN_V2, DINOV1
except ImportError:
    DINO_DQN_V1 = os.path.join(PROJECT_ROOT, "models", "dino_dqn_v1.zip")
    DINO_DQN_V2 = os.path.join(PROJECT_ROOT, "models", "dino_dqn_v2.zip")
    DINOV1 = os.path.join(PROJECT_ROOT, "models", "dinoV1.zip")

__all__ = [
    "DINO_DQN_V1",
    "DINO_DQN_V2",
    "DINOV1",
]
