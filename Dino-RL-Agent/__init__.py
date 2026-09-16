"""
Dino RL Agent: Autonomous Chrome Dino Reinforcement Learning System.

This package provides:
- Vision pipeline: screen capture, preprocessing, object detection, and perception
- Environments: Gymnasium-compatible DinoEnv (v1) and DinoEnvV2 (calibrated v2)
- Controllers: PyAutoGUI and keyboard action execution with failsafe handling
- Agents: Stable-Baselines3 DQN model definitions, training, and execution
"""

__version__ = "1.0.0"

from .src import (
    # Vision
    VideoRecorder,
    ScreenCapture,
    Preprocessor,
    Detector,
    Observation,
    ObservationManager,
    # Environments
    DinoEnv,
    DinoEnvV1,
    DinoEnvV2,
    # Controllers
    BaseDinoController,
    EnvV1Controller,
    DinoV1Controller,
    EnvV2Controller,
    DinoV2Controller,
    # Agents & Configurations
    dinoV1,
    DINO_DQN_V1,
    DINO_DQN_V2,
    DINOV1,
    # Configuration & runners
    MODEL_PATH,
    GAME_REGION,
    DINO_ROI_WIDTH,
    run_agent,
    run_dinoV1,
)

__all__ = [
    "__version__",
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