"""
Gymnasium simulation environments for Chrome Dino reinforcement learning.

Environments:
    - DinoEnv: Version 1 environment with a 5-element observation vector.
    - DinoEnvV1: Alias for DinoEnv.
    - DinoEnvV2: Version 2 environment calibrated against real vision telemetry,
                 featuring a 7-element state vector and relative velocity physics.
"""
from .dino_env_v1 import DinoEnv
from .dino_env_v2 import DinoEnvV2

DinoEnvV1 = DinoEnv

__all__ = [
    "DinoEnv", 
    "DinoEnvV1",
    "DinoEnvV2",
]
