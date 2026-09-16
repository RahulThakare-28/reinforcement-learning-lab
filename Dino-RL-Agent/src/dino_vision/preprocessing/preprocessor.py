import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple


class Preprocessor:
    """
    Preprocess captured Chrome Dino frames into clean 1-channel representations
    (grayscale and binary black-and-white) for object detection and reinforcement learning.
    """

    def __init__(self, threshold_value: int = 128, auto_invert: bool = True):
        """
        Parameters
        ----------
        threshold_value : int
            Pixel intensity cutoff for binary thresholding (default: 128).
        auto_invert : bool
            Automatically detect daytime vs nighttime mode in the Dino game
            to guarantee foreground objects are always WHITE (255) and background is BLACK (0).
        """
        self.threshold_value = threshold_value
        self.auto_invert = auto_invert

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Convert the raw captured screen frame into clean 1-channel representations.

        Parameters
        ----------
        frame : np.ndarray
            Captured screen frame (RGB, BGR, RGBA, or Grayscale).

        Returns
        -------
        dict
            {
                "original": raw input frame,
                "gray": 1-channel grayscale image (shape: H x W),
                "binary": 1-channel black-white binary mask (shape: H x W, foreground=255, background=0),
                "is_night_mode": bool indicating if dark theme was detected
            }
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        # 1. Convert input to 1-channel Grayscale based on input shape
        if len(frame.shape) == 3:
            channels = frame.shape[2]
            if channels == 4:
                # RGBA -> Gray
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                # RGB -> Gray (pyautogui screenshots are RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif channels == 1:
                gray = frame.squeeze(axis=2)
            else:
                gray = frame[:, :, 0]
        elif len(frame.shape) == 2:
            gray = frame.copy()
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        # 2. Polarity Detection (Day vs Night mode in Chrome Dino)
        # In Chrome Dino:
        # - Daytime mode: Background is light (~245-255), objects (dino, cacti, birds) are dark (~83)
        # - Nighttime mode: Background is dark (~32), objects are light (~172)
        mean_intensity = float(gray.mean())
        is_night_mode = mean_intensity < self.threshold_value

        # 3. Binary 1-channel thresholding
        # Goal: Foreground objects must ALWAYS be WHITE (255), background BLACK (0) for detector contours
        if self.auto_invert:
            if is_night_mode:
                # Dark background -> Foreground objects are lighter than background
                _, binary = cv2.threshold(
                    gray,
                    self.threshold_value,
                    255,
                    cv2.THRESH_BINARY
                )
            else:
                # Light background -> Invert so dark foreground objects become white (255)
                _, binary = cv2.threshold(
                    gray,
                    self.threshold_value,
                    255,
                    cv2.THRESH_BINARY_INV
                )
        else:
            _, binary = cv2.threshold(
                gray,
                self.threshold_value,
                255,
                cv2.THRESH_BINARY
            )

        return {
            "original": frame,
            "gray": gray,
            "binary": binary,
            "is_night_mode": is_night_mode,
        }


# =====================================================================
# Interactive Pipeline Test: ScreenCapture -> Preprocessor
# =====================================================================
if __name__ == "__main__":
    import sys
    import os

    # Ensure src and project root are in sys.path (safe for both .py scripts and Jupyter Notebooks)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
    except NameError:
        # Running inside Jupyter Notebook or interactive REPL where __file__ is undefined
        from pathlib import Path
        cwd = Path.cwd().resolve()
        if cwd.name == "notebooks":
            src_path = str(cwd.parent / "src")
            project_root = str(cwd.parent)
        elif (cwd / "src").is_dir():
            src_path = str(cwd / "src")
            project_root = str(cwd)
        else:
            src_path = str(cwd)
            project_root = str(cwd)

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from dino_vision.capture.screen_capture import ScreenCapture
    except ImportError:
        from src.dino_vision.capture.screen_capture import ScreenCapture

    print("=" * 70)
    print("Dino Vision: Screen Capture -> 1-Channel Preprocessor Pipeline")
    print("=" * 70)
    print("Pipeline Flow:")
    print("  [ScreenCapture] -> RGB Frame -> [Preprocessor] -> 1-Channel Binary (B&W)")
    print("Controls:")
    print("  - Use the transparent capture window to target Chrome Dino.")
    print("  - Preview window shows:")
    print("      Top / Left : Live Color Capture")
    print("      Bottom/Right: 1-Channel Black & White Foreground Mask (255=Objects, 0=Background)")
    print("  - Press 'q' or ESC in preview window to exit cleanly.")
    print("=" * 70)

    # Instantiate preprocessor first BEFORE opening GUI window
    preprocessor = Preprocessor(threshold_value=128, auto_invert=True)
    capture = ScreenCapture()
    window_name = "Dino Vision: Live Capture vs 1-Channel B&W"

    try:
        capture.select_region()
        capture.play()

        for frame in capture.start(fps_limit=30):
            # Process captured frame into 1-channel representations
            data = preprocessor.process(frame)
            binary_1ch = data["binary"]
            is_night = data["is_night_mode"]

            # Convert 1-channel binary to 3-channel for side-by-side visualization
            binary_vis = cv2.cvtColor(binary_1ch, cv2.COLOR_GRAY2BGR)
            color_vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Match heights if needed and stack side by side
            h, w = color_vis.shape[:2]
            if binary_vis.shape[:2] != (h, w):
                binary_vis = cv2.resize(binary_vis, (w, h))

            # Overlay labels
            mode_text = "Night Mode" if is_night else "Day Mode"
            cv2.putText(color_vis, f"Raw Screen ({mode_text})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(binary_vis, "1-Channel Binary (Foreground=White)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            combined_preview = np.hstack([color_vis, binary_vis])
            cv2.imshow(window_name, combined_preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                capture.close()
                break

    except KeyboardInterrupt:
        pass
    finally:
        capture.close()
        cv2.destroyAllWindows()
        print("[Done] Pipeline test completed cleanly.")