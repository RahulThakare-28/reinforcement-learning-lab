"""
DinoV1 Agent Control: Vision Pipeline + SB3 DQN Agent + EnvV2Controller

Perception -> Prediction -> Action Execution:
1. ScreenCapture grabs game frames.
2. Preprocessor produces binary frame (bg=WHITE, objects=BLACK).
3. Detector detects Dino and Obstacles.
4. ObservationManager computes the 7-element observation vector:
   [distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type]
5. Trained DQN model (dinoV1.zip) predicts discrete action:
   0 = DUCK, 1 = JUMP, 2 = DO NOTHING
6. EnvV2Controller sends corresponding keypresses (DOWN, UP, or NONE).
"""

import os
import sys
import time
import logging
from typing import Optional

# ── Path resolution: supports direct execution and package imports ────────────
try:
    _here = os.path.abspath(os.path.dirname(__file__))
except NameError:
    from pathlib import Path
    _cwd = Path.cwd().resolve()
    _here = str(_cwd)

# Adjust hops to 'src' (2 steps up) and 'Dino-RL-Agent' project root (3 steps up)
SRC_PATH = os.path.abspath(os.path.join(_here, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", "..", ".."))

for p in [PROJECT_ROOT, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
from stable_baselines3 import DQN

# ── Import agents config ──────────────────────────────────────────────────────
try:
    from agents_config import DINOV1
except (ImportError, ValueError):
    try:
        from src.agents.agents_config import DINOV1
    except ImportError:
        from agents_config import DINOV1

# ── Import Vision and Control modules ─────────────────────────────────────────
try:
    from controls.env_v2_control import EnvV2Controller
    from dino_vision.capture.screen_capture import ScreenCapture
    from dino_vision.preprocessing.preprocessor import Preprocessor
    from dino_vision.detection.detector import Detector
    from dino_vision.perception.observation import ObservationManager
except ImportError:
    from src.controls.env_v2_control import EnvV2Controller
    from src.dino_vision.capture.screen_capture import ScreenCapture
    from src.dino_vision.preprocessing.preprocessor import Preprocessor
    from src.dino_vision.detection.detector import Detector
    from src.dino_vision.perception.observation import ObservationManager

logger = logging.getLogger("DinoV1.Control")


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolve the DQN model path to an existing absolute file path."""
    candidate_path = model_path or DINOV1

    # If candidate is already absolute and exists
    if os.path.isabs(candidate_path) and os.path.exists(candidate_path):
        return candidate_path

    # Try relative to PROJECT_ROOT
    root_candidate = os.path.join(PROJECT_ROOT, candidate_path)
    if os.path.exists(root_candidate):
        return root_candidate

    # Try relative to models directory in PROJECT_ROOT
    model_name = os.path.basename(candidate_path)
    models_dir_candidate = os.path.join(PROJECT_ROOT, "models", model_name)
    if os.path.exists(models_dir_candidate):
        return models_dir_candidate

    # Try relative to _here
    here_candidate = os.path.abspath(os.path.join(_here, candidate_path))
    if os.path.exists(here_candidate):
        return here_candidate

    # Fallback to candidate_path as given
    return candidate_path


def run_dinoV1(
    model_path: Optional[str] = None,
    show_hud: bool = True,
    fps_limit: int = 30,
    log_obs: bool = False,
    pause_on_start: bool = True
) -> None:
    """
    Run the trained DinoV1 agent with real-time vision observation and keyboard action execution.

    Parameters:
        model_path (str, optional): Path to the trained dinoV1 DQN .zip model.
        show_hud (bool): Whether to show OpenCV visualization HUD window.
        fps_limit (int): Frame capture rate limit (default: 30).
        log_obs (bool): Whether to log observations into the Obs/ folder.
        pause_on_start (bool): Start in paused state until 's' or 'r' is pressed.
    """
    resolved_path = resolve_model_path(model_path)
    print("=" * 70)
    print("           DINO V1 RL AGENT (VISION + DQN + V2 CONTROL)")
    print("=" * 70)
    print(f"[INFO] Loading model from: {resolved_path}")

    try:
        model = DQN.load(resolved_path)
        print(f"[OK] DinoV1 model loaded successfully.")
        print(f"     Observation Space: {model.observation_space}")
        print(f"     Action Space     : {model.action_space}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from '{resolved_path}': {e}")
        return

    # ── Initialize components ─────────────────────────────────────────────────
    controller = EnvV2Controller(failsafe=True, pause_on_start=pause_on_start)
    preprocessor = Preprocessor(auto_invert=True)
    detector = Detector()
    obs_manager = ObservationManager()
    capture = ScreenCapture()
    window_name = "DinoV1 RL Agent - Vision & Control HUD"

    # Start keyboard listener
    controller.start_controller()

    print("\nKeyboard Hotkeys:")
    print("  's' - Start agent")
    print("  'p' - Pause agent")
    print("  'r' - Resume agent")
    print("  'q' - Quit agent (or 'q' / ESC in HUD window)")
    print("=" * 70)

    # ── Screen Region Selection ───────────────────────────────────────────────
    print("\n[INFO] Please select the Chrome Dino game window region...")
    try:
        capture.select_region()
        capture.play()
    except Exception as e:
        print(f"[ERROR] Screen region selection failed: {e}")
        controller.stop_controller()
        return

    frame_count = 0
    action_names = {0: "DUCK", 1: "JUMP", 2: "DO_NOTHING"}

    print("\n[INFO] Perception & Control loop active.")
    if controller.paused:
        print("[STATUS] Agent is currently PAUSED. Press 's' to START playing.\n")
    else:
        print("[STATUS] Agent is RUNNING.\n")

    try:
        for frame in capture.start(fps_limit=fps_limit):
            # Check quit flag
            if controller.quit_game:
                print("\n[INFO] Quit signal received from controller.")
                break

            # Check pause flag
            if controller.paused:
                if show_hud:
                    # Keep HUD responsive even when paused
                    try:
                        prep_data = preprocessor.process(frame)
                        vis = cv2.cvtColor(prep_data["binary"], cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 150), -1)
                        cv2.putText(
                            vis,
                            "AGENT PAUSED - Press 's' to start or 'r' to resume",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                        cv2.imshow(window_name, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            frame_count += 1
            now = time.time()

            # ── 1. Preprocessing ──────────────────────────────────────────────
            try:
                prep_data = preprocessor.process(frame)
                binary = prep_data["binary"]
            except Exception as e:
                logger.error(f"Preprocessor error on frame {frame_count}: {e}")
                continue

            # ── 2. Detection ──────────────────────────────────────────────────
            try:
                detections = detector.detect(binary)
                if not isinstance(detections, dict):
                    continue
            except Exception as e:
                logger.error(f"Detector error on frame {frame_count}: {e}")
                continue

            # ── 3. Observation Vector Formulation ─────────────────────────────
            try:
                # Returns 7-element float32 numpy vector
                obs_vector = obs_manager.get_observation(detections, current_time=now)
                observation = obs_manager.current_observation
                obs_dict = observation.to_dict() if observation else {}
            except Exception as e:
                logger.error(f"ObservationManager error on frame {frame_count}: {e}")
                continue

            # Optional JSON logging
            if log_obs:
                obs_manager.log_observation(observation, frame_id=frame_count)

            # ── 4. Model Prediction ───────────────────────────────────────────
            try:
                action, _ = model.predict(obs_vector, deterministic=True)
                action = int(action)
            except Exception as e:
                logger.error(f"DQN prediction error on frame {frame_count}: {e}")
                continue

            # ── 5. Action Execution via EnvV2Controller ────────────────────────
            # Pass is_jumping so Fix 3 cooldown can suppress repeated JUMP presses
            is_jumping = bool(obs_dict.get("dino", {}).get("is_jumping", False))
            controller.execute_action(action, is_jumping=is_jumping)

            # ── Terminal telemetry ────────────────────────────────────────────
            if frame_count % 15 == 0 or action != 2:
                act_str = action_names.get(action, str(action))
                print(
                    f"[Frame {frame_count:04d}] Action: {act_str:10s} | "
                    f"Dist: {obs_vector[0]:5.1f} | Vel: {obs_vector[1]:4.1f} | "
                    f"Type: {obs_vector[6]:.0f} | Obs: {obs_vector.round(1).tolist()}"
                )

            # ── 6. HUD Visualization Overlay ───────────────────────────────────
            if show_hud:
                try:
                    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    dino = obs_dict.get("dino", {})
                    obs = obs_dict.get("obstacle", {})

                    # Dino bounding box
                    if isinstance(dino, dict) and dino.get("width", 0) > 0:
                        cv2.rectangle(
                            vis,
                            (dino["x"], dino["y"]),
                            (dino["x"] + dino["width"], dino["y"] + dino["height"]),
                            (0, 255, 0),
                            2
                        )
                        jump_lbl = "JUMPING" if dino.get("is_jumping") else "GROUND"
                        cv2.putText(
                            vis,
                            jump_lbl,
                            (dino["x"], max(0, dino["y"] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 220, 255),
                            1
                        )

                    # Obstacle bounding box
                    if isinstance(obs, dict) and obs.get("width", 0) > 0:
                        color = (0, 0, 255) if obs.get("type") == "cactus" else (255, 120, 0)
                        cv2.rectangle(
                            vis,
                            (obs["x"], obs["y"]),
                            (obs["x"] + obs["width"], obs["y"] + obs["height"]),
                            color,
                            2
                        )
                        obs_label = f"{str(obs.get('type', '')).upper()} ({obs_vector[6]:.0f})"
                        cv2.putText(
                            vis,
                            obs_label,
                            (obs["x"], max(0, obs["y"] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1
                        )

                        # Distance line
                        if isinstance(dino, dict) and dino.get("width", 0) > 0:
                            cv2.line(
                                vis,
                                (dino["x"] + dino["width"], dino["y"] + dino["height"] // 2),
                                (obs["x"], obs["y"] + obs["height"] // 2),
                                (255, 0, 255),
                                2
                            )
                            mid_x = (dino["x"] + dino["width"] + obs["x"]) // 2
                            cv2.putText(
                                vis,
                                f"Dist: {obs_vector[0]:.0f}px",
                                (max(10, mid_x - 40), max(20, dino["y"] - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 0, 255),
                                1
                            )

                    # Status bar
                    action_color = (0, 255, 255) if action == 2 else ((0, 255, 0) if action == 1 else (0, 165, 255))
                    status_text = (
                        f"ACT: {action_names.get(action, str(action))} | "
                        f"Dist: {obs_vector[0]:.0f} | Vel: {obs_vector[1]:.1f} | "
                        f"Type: {obs_dict.get('type', 'none')}"
                    )
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (30, 30, 30), -1)
                    cv2.putText(
                        vis,
                        status_text,
                        (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.46,
                        action_color,
                        1
                    )

                    cv2.imshow(window_name, vis)

                except Exception as ve:
                    logger.warning(f"Visualization error on frame {frame_count}: {ve}")

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\n[INFO] HUD quit requested by user.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Agent interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in agent loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture.close()
        cv2.destroyAllWindows()
        controller.stop_controller()
        print("[DONE] DinoV1 agent session completed cleanly.")


if __name__ == "__main__":
    run_dinoV1()
