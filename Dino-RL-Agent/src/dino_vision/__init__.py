"""
Dino Vision: End-to-end computer vision and perception pipeline for Chrome Dino RL.

Subpackages:
    - capture: Screen frame grabbing and video recording
    - preprocessing: Frame cleaning, thresholding, and night-mode detection
    - detection: Dino and obstacle localization
    - perception: State vector generation and kinematics tracking
"""
from .capture import VideoRecorder, ScreenCapture
from .preprocessing import Preprocessor
from .detection import Detector
from .perception import Observation, ObservationManager

__all__ = [
    "VideoRecorder",
    "ScreenCapture",
    "Preprocessor",
    "Detector",
    "Observation",
    "ObservationManager",
]