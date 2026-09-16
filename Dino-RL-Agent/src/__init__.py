"""
Chrome Dino Reinforcement Learning Lab (src package).

Exposes core modules and subsystems:
- dino_vision: Screen capture, preprocessing, object detection, and observation pipeline
- dino_env: Gymnasium training environments (DinoEnv, DinoEnvV2)
- controls: Input listeners and keyboard action executors
- agents: DQN model builders, training routines, and execution
"""
from .dino_vision import (
    VideoRecorder,
    ScreenCapture,
    Preprocessor,
    Detector,
    Observation,
    ObservationManager,
)
from .dino_env import DinoEnv, DinoEnvV1, DinoEnvV2
from .controls import (
    BaseDinoController,
    EnvV1Controller,
    DinoV1Controller,
    EnvV2Controller,
    DinoV2Controller,
)
from .agents import dinoV1, DINO_DQN_V1, DINO_DQN_V2, DINOV1
from .config import MODEL_PATH, GAME_REGION, DINO_ROI_WIDTH


def run_agent(*args, **kwargs):
    """
    Run the trained Dino RL agent.
    Dynamically imported from agent2 to avoid side effects during package initialization.
    """
    from .agent2 import run_agent as _run_agent
    return _run_agent(*args, **kwargs)


def run_dinoV1(*args, **kwargs):
    """
    Run the trained DinoV1 agent with live vision observation and EnvV2Controller.
    Dynamically imported to avoid side effects during package initialization.
    """
    from .agents.dinoV1 import run_dinoV1 as _run_dinoV1
    return _run_dinoV1(*args, **kwargs)


__all__ = [
    # Vision pipeline
    "VideoRecorder",
    "ScreenCapture",
    "Preprocessor",
    "Detector",
    "Observation",
    "ObservationManager",
    # Environments
    "DinoEnv",
    "DinoEnvV1",
    "DinoEnvV2",
    # Controllers
    "BaseDinoController",
    "EnvV1Controller",
    "DinoV1Controller",
    "EnvV2Controller",
    "DinoV2Controller",
    # Agents
    "dinoV1",
    "DINO_DQN_V1",
    "DINO_DQN_V2",
    "DINOV1",
    # Configuration
    "MODEL_PATH",
    "GAME_REGION",
    "DINO_ROI_WIDTH",
    # Agent runners
    "run_agent",
    "run_dinoV1",
]