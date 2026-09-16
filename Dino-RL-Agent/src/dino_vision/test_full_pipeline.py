"""
Dino Vision: Full Pipeline Test
ScreenCapture -> Preprocessor -> Detector -> Observation Layer

Executes the complete vision perception pipeline for the Chrome Dino RL agent.
- Produces the 7-element numerical observation vector via ObservationManager.get_observation()
- Computes approaching relative velocity (v_rel) between Dino and obstacles
- Classifies obstacle type: 0 for Bird, 1 for Cactus (-1 for none)
- Prints the observation vector on the terminal
- Logs capture data into JSON format in the Obs/ folder
"""
import sys
import os
import time
import json

# ── Path resolution: works as .py script AND in Jupyter notebook ──────────────
try:
    _here = os.path.abspath(os.path.dirname(__file__))
except NameError:
    from pathlib import Path
    _cwd = Path.cwd().resolve()
    _here = str(_cwd)

PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
SRC_PATH = os.path.abspath(os.path.join(_here, ".."))

for p in [PROJECT_ROOT, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

try:
    from dino_vision.capture.screen_capture import ScreenCapture
    from dino_vision.preprocessing.preprocessor import Preprocessor
    from dino_vision.detection.detector import Detector
    from dino_vision.perception.observation import ObservationManager
    print("[OK] All dino_vision modules imported (dino_vision.* prefix).")
except ImportError:
    from src.dino_vision.capture.screen_capture import ScreenCapture
    from src.dino_vision.preprocessing.preprocessor import Preprocessor
    from src.dino_vision.detection.detector import Detector
    from src.dino_vision.perception.observation import ObservationManager
    print("[OK] All dino_vision modules imported (src.dino_vision.* prefix).")

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 75)
print("Dino Vision: ScreenCapture -> Preprocessor -> Detector -> Observation")
print("=" * 75)
print("Pipeline Flow:")
print("  [ScreenCapture] -> RGB Frame")
print("  -> [Preprocessor] -> 1-Channel Binary (bg=WHITE, objects=BLACK)")
print("  -> [Detector]     -> {dino, obstacle, ground_y}")
print("  -> [Observation]  -> get_observation() [7-Element Vector]")
print()
print("Vector Format: (distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type)")
print("Type Coding  : 0 = Bird, 1 = Cactus, -1 = None")
print()
print("Controls (OpenCV window must be focused):")
print("  'q' / ESC - quit cleanly")
print("  Capture window: [Play]  [Pause]  [Cancel]")
print("=" * 75)

# ── Instantiate pipeline layers ───────────────────────────────────────────────
preprocessor = Preprocessor(auto_invert=True)
detector = Detector()
obs_manager = ObservationManager()
capture = ScreenCapture()
window_name = "Dino Vision - Agent Observation HUD"

print(f"[INFO] JSON observations will be saved to: {obs_manager.obs_dir}")

# ── Run pipeline ──────────────────────────────────────────────────────────────
try:
    capture.select_region()
    capture.play()

    frame_count = 0
    print("\n[INFO] Streaming started. Observations printed to terminal. Press 'q' to quit.\n")

    for frame in capture.start(fps_limit=30):
        frame_count += 1
        now = time.time()

        # ── 1. Preprocess ──────────────────────────────────────────────────────
        try:
            prep_data = preprocessor.process(frame)
            binary = prep_data["binary"]
        except Exception as e:
            print(f"[ERROR] Preprocessor failed on frame {frame_count}: {e}")
            continue

        # ── 2. Detect ──────────────────────────────────────────────────────────
        try:
            detections = detector.detect(binary)
            if not isinstance(detections, dict):
                print(
                    f"[ERROR] detector.detect() returned {type(detections).__name__} "
                    f"on frame {frame_count}. Expected dict. Skipping."
                )
                continue
        except Exception as e:
            print(f"[ERROR] Detector failed on frame {frame_count}: {e}")
            continue

        # ── 3. Formulate Observation Vector ────────────────────────────────────
        try:
            # Calls get_observation() which returns (7,) float32 vector
            obs_vector = obs_manager.get_observation(detections, current_time=now)
            observation = obs_manager.current_observation
            obs_dict = observation.to_dict() if observation else {}
        except Exception as e:
            print(f"[ERROR] ObservationManager failed on frame {frame_count}: {e}")
            continue

        # ── Log to JSON in Obs/ folder on every frame ───────────────────────────
        obs_manager.log_observation(observation, frame_id=frame_count)

        # ── Print observation vector on terminal every 15 frames ───────────────
        if frame_count % 15 == 0:
            print(f"\n[Frame {frame_count:04d}] --- Observation Vector for Agent ---")
            print(f"  Vector: {obs_vector.round(2).tolist()}")
            print(
                f"  Labels: dist={obs_vector[0]:.1f} | vel={obs_vector[1]:.1f} | "
                f"dino=({obs_vector[3]:.0f}x{obs_vector[2]:.0f}) | "
                f"obs=({obs_vector[5]:.0f}x{obs_vector[4]:.0f}) | type={obs_vector[6]:.0f}"
            )

        # ── 4. HUD Visualization Overlay ───────────────────────────────────────
        try:
            vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            dino = obs_dict.get("dino", {})
            obs = obs_dict.get("obstacle", {})

            # ── Dino: green box + velocity + jump state ───────────────────────
            if isinstance(dino, dict) and dino.get("width", 0) > 0:
                cv2.rectangle(
                    vis,
                    (dino["x"], dino["y"]),
                    (dino["x"] + dino["width"], dino["y"] + dino["height"]),
                    (0, 255, 0), 2
                )
                jump_lbl = "JUMPING" if dino.get("is_jumping") else "GROUND"
                cv2.putText(
                    vis, jump_lbl,
                    (dino["x"], max(0, dino["y"] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1
                )

            # ── Obstacle: red (cactus) or blue (bird) + distance line ─────────
            if isinstance(obs, dict) and obs.get("width", 0) > 0:
                color = (0, 0, 255) if obs.get("type") == "cactus" else (255, 120, 0)
                cv2.rectangle(
                    vis,
                    (obs["x"], obs["y"]),
                    (obs["x"] + obs["width"], obs["y"] + obs["height"]),
                    color, 2
                )
                obs_label = f"{str(obs.get('type', '')).upper()} (code={obs_vector[6]:.0f})"
                cv2.putText(
                    vis, obs_label,
                    (obs["x"], max(0, obs["y"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                )

                if isinstance(dino, dict) and dino.get("width", 0) > 0:
                    cv2.line(
                        vis,
                        (dino["x"] + dino["width"], dino["y"] + dino["height"] // 2),
                        (obs["x"], obs["y"] + obs["height"] // 2),
                        (255, 0, 255), 2
                    )
                    mid_x = (dino["x"] + dino["width"] + obs["x"]) // 2
                    cv2.putText(
                        vis,
                        f"Dist: {obs_vector[0]:.0f}px | V_rel: {obs_vector[1]:.1f}",
                        (max(10, mid_x - 60), max(20, dino["y"] - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
                    )

            # ── HUD top bar: all 7 vector elements ────────────────────────────
            hud_text = (
                f"Dist:{obs_vector[0]:.0f} | "
                f"Vel:{obs_vector[1]:.1f} | "
                f"Dino:({obs_vector[3]:.0f}x{obs_vector[2]:.0f}) | "
                f"Obs:({obs_vector[5]:.0f}x{obs_vector[4]:.0f}) | "
                f"Type:{obs_dict.get('type', 'none')}({obs_vector[6]:.0f})"
            )
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (30, 30, 30), -1)
            cv2.putText(vis, hud_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

            cv2.imshow(window_name, vis)

        except Exception as ve:
            print(f"[WARN] Visualization error on frame {frame_count}: {ve}")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")
except Exception as e:
    print(f"\n[ERROR] Unexpected pipeline error: {e}")
    import traceback
    traceback.print_exc()
finally:
    capture.close()
    cv2.destroyAllWindows()
    print(f"[Done] Full pipeline observation test finished cleanly. Logs saved to: {obs_manager.obs_dir}")
