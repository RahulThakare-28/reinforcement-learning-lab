import cv2
import numpy as np


class Preprocessor:
    """
    Preprocess captured Chrome Dino frames
    before they are passed to the detection layer.
    """

    def __init__(self):
        pass

    def process(self, frame: np.ndarray) -> dict:
        """
        Convert the captured frame into useful
        representations for the detection layer.

        Parameters
        ----------
        frame : np.ndarray
            Raw screenshot frame.

        Returns
        -------
        dict
            Preprocessed image representations.
        """

        if frame is None:
            raise ValueError("Input frame is None.")

        # 1. Convert RGB screenshot to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 2. Convert grayscale image to binary
        _, binary = cv2.threshold(
            gray,
            128,
            255,
            cv2.THRESH_BINARY
        )

        return {
            "original": frame,
            "gray": gray,
            "binary": binary,
        }