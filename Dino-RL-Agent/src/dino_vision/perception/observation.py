import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("DinoVision.Observation")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [Observation] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Observation:
    """
    Data container representing the processed state observation for the RL agent.
    """

    def __init__(
        self,
        dino: Optional[Dict[str, Any]] = None,
        obstacle: Optional[Dict[str, Any]] = None,
        distance: float = 500.0,
        motion: float = 0.0,
        velocity: float = 0.0,
        height: float = 0.0,
        width: float = 0.0,
        obstacle_type: str = "none",
        obstacle_type_code: int = 0,
    ):
        self.dino = dino or {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "velocity": 0.0,
            "is_jumping": False,
        }
        self.obstacle = obstacle or {
            "type": "none",
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "distance": distance,
        }
        self.distance = float(distance)
        self.motion = float(motion)
        self.velocity = float(velocity)
        self.height = float(height)
        self.width = float(width)
        self.type = str(obstacle_type)
        self.type_code = int(obstacle_type_code)

    def to_dict(self) -> Dict[str, Any]:
        """
        Produce the structured observation object requested for the agent.
        """
        return {
            "dino": {
                "x": self.dino.get("x", 0),
                "y": self.dino.get("y", 0),
                "width": self.dino.get("width", 0),
                "height": self.dino.get("height", 0),
                "velocity": self.dino.get("velocity", self.velocity),
                "is_jumping": self.dino.get("is_jumping", False),
            },
            "obstacle": {
                "type": self.obstacle.get("type", self.type),
                "x": self.obstacle.get("x", 0),
                "y": self.obstacle.get("y", 0),
                "width": self.obstacle.get("width", int(self.width)),
                "height": self.obstacle.get("height", int(self.height)),
                "distance": self.distance,
            },
            "distance": self.distance,
            "motion": self.motion,
            "velocity": self.velocity,
            "height": self.height,
            "width": self.width,
            "type": self.type,
        }

    def to_array(self) -> np.ndarray:
        """
        Produce a normalized/raw 1D NumPy array compatible with RL models (Gymnasium / SB3).
        [distance, obstacle_type_code, obstacle_height, obstacle_width, dino_height, dino_velocity]
        """
        dino_y = float(self.dino.get("y", 0.0))
        return np.array(
            [
                self.distance,
                float(self.type_code),
                self.height,
                self.width,
                dino_y,
                self.velocity,
            ],
            dtype=np.float32,
        )

    def __repr__(self) -> str:
        return (
            f"Observation(distance={self.distance:.1f}, "
            f"type='{self.type}', "
            f"height={self.height:.1f}, "
            f"width={self.width:.1f}, "
            f"velocity={self.velocity:.1f}, "
            f"motion={self.motion:.1f})"
        )


class ObservationManager:
    """
    Observation Layer responsible for computing:
    - Distance (between Dino and incoming obstacle)
    - Motion & Velocity (Dino jump / fall dynamics over time)
    - Height & Width (obstacle size)
    - Type ("cactus" vs "bird")
    """

    def __init__(self, max_distance: float = 600.0, ground_y_baseline: int = 216):
        self.max_distance = max_distance
        self.ground_y_baseline = ground_y_baseline
        self.previous_dino_y: Optional[float] = None
        self.previous_time: Optional[float] = None

    def reset(self) -> None:
        """Reset velocity and motion tracking state for new game / episode."""
        self.previous_dino_y = None
        self.previous_time = None
        logger.info("ObservationManager state reset.")

    def calculate_distance(
        self, dino: Optional[Dict[str, Any]], obstacle: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate horizontal clearance between Dino and obstacle."""
        try:
            if not dino or not obstacle:
                return float(self.max_distance)

            dino_front_x = float(dino["x"] + dino["width"])
            obstacle_front_x = float(obstacle["x"])

            distance = obstacle_front_x - dino_front_x
            return float(max(0.0, distance))
        except Exception as e:
            logger.warning(f"Error calculating distance: {e}")
            return float(self.max_distance)

    def calculate_dino_motion(
        self, dino: Optional[Dict[str, Any]]
    ) -> Tuple[float, float, bool]:
        """
        Calculate Dino's current vertical position (motion),
        vertical velocity (previous_y - current_y), and jumping flag.
        """
        try:
            if not dino:
                return 0.0, 0.0, False

            current_y = float(dino["y"])

            if self.previous_dino_y is None:
                velocity = 0.0
            else:
                # Positive velocity = ascending (jumping up)
                # Negative velocity = descending (falling down)
                velocity = self.previous_dino_y - current_y

            self.previous_dino_y = current_y

            # Check if dino is jumping (airborne above baseline ground)
            is_jumping = (current_y + dino["height"]) < (self.ground_y_baseline - 10)

            return current_y, float(velocity), is_jumping
        except Exception as e:
            logger.warning(f"Error calculating dino motion: {e}")
            return 0.0, 0.0, False

    def create_observation(self, detection_result: Dict[str, Any]) -> Observation:
        """
        Transform detection layer output into an Observation object.

        Parameters
        ----------
        detection_result : dict
            Output from Detector.detect(), containing "dino", "obstacle", and "ground_y".

        Returns
        -------
        Observation
            Clean, structured observation object.
        """
        try:
            dino_data = detection_result.get("dino")
            obstacle_data = detection_result.get("obstacle")
            ground_y = detection_result.get("ground_y", self.ground_y_baseline)

            # 1. Distance
            distance = self.calculate_distance(dino_data, obstacle_data)

            # 2. Motion & Velocity
            motion, velocity, is_jumping = self.calculate_dino_motion(dino_data)

            # Enrich dino dictionary
            enriched_dino = None
            if dino_data:
                enriched_dino = {
                    "x": int(dino_data["x"]),
                    "y": int(dino_data["y"]),
                    "width": int(dino_data["width"]),
                    "height": int(dino_data["height"]),
                    "velocity": float(velocity),
                    "is_jumping": bool(is_jumping),
                }

            # 3. Obstacle Dimensions & Type
            if obstacle_data:
                obs_type = str(obstacle_data.get("type", "cactus")).lower()
                obs_width = float(obstacle_data.get("width", 0.0))
                obs_height = float(obstacle_data.get("height", 0.0))
                type_code = 1 if obs_type == "bird" else 0

                enriched_obstacle = {
                    "type": obs_type,
                    "x": int(obstacle_data.get("x", 0)),
                    "y": int(obstacle_data.get("y", 0)),
                    "width": int(obs_width),
                    "height": int(obs_height),
                    "distance": float(distance),
                }
            else:
                obs_type = "none"
                obs_width = 0.0
                obs_height = 0.0
                type_code = 0
                enriched_obstacle = {
                    "type": "none",
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0,
                    "distance": float(distance),
                }

            return Observation(
                dino=enriched_dino,
                obstacle=enriched_obstacle,
                distance=distance,
                motion=motion,
                velocity=velocity,
                height=obs_height,
                width=obs_width,
                obstacle_type=obs_type,
                obstacle_type_code=type_code,
            )

        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            # Fallback safe observation to prevent crashing the agent
            return Observation(
                distance=self.max_distance,
                motion=0.0,
                velocity=0.0,
                height=0.0,
                width=0.0,
                obstacle_type="none",
                obstacle_type_code=0,
            )


# =====================================================================
# Interactive Diagnostic & Testing
# =====================================================================
if __name__ == "__main__":
    import sys
    import os
    import json
    import time
    import cv2

    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from dino_vision.capture.screen_capture     import ScreenCapture
    from dino_vision.preprocessing.preprocessor import Preprocessor
    from dino_vision.detection.detector          import Detector

    print("=" * 70)
    print("Dino Vision: ScreenCapture -> Preprocessor -> Detector -> Observation")
    print("=" * 70)
    print("Pipeline Flow:")
    print("  [ScreenCapture] -> RGB Frame")
    print("  -> [Preprocessor] -> 1-Channel Binary")
    print("  -> [Detector]     -> {dino, obstacle, ground_y}")
    print("  -> [Observation]  -> Agent Observation Object")
    print()
    print("Controls (OpenCV window must be focused):")
    print("  'q' / ESC — quit cleanly")
    print("  Capture window: [▶ Play] [⏸ Pause] [✕ Cancel]")
    print("=" * 70)

    preprocessor = Preprocessor(auto_invert=True)
    detector     = Detector()
    obs_manager  = ObservationManager()
    capture      = ScreenCapture()
    window_name  = "Dino Vision - Agent Observation HUD"

    try:
        capture.select_region()
        capture.play()

        frame_count = 0
        print("[INFO] Streaming frames and generating agent observations. Press 'q' to quit.\n")

        for frame in capture.start(fps_limit=30):
            frame_count += 1

            # ── 1. Preprocess ──────────────────────────────────────────────────
            try:
                prep_data = preprocessor.process(frame)
                binary    = prep_data["binary"]
            except Exception as e:
                logger.error(f"Preprocessor failed on frame {frame_count}: {e}")
                continue

            # ── 2. Detect ──────────────────────────────────────────────────────
            try:
                detections = detector.detect(binary)
                if not isinstance(detections, dict):
                    logger.error(
                        f"detector.detect() returned unexpected type "
                        f"'{type(detections).__name__}' on frame {frame_count}. "
                        "Expected dict. Skipping frame. "
                        "(Restart the kernel if this persists.)"
                    )
                    continue
            except Exception as e:
                logger.error(f"Detector failed on frame {frame_count}: {e}")
                continue

            # ── 3. Formulate Observation ───────────────────────────────────────
            try:
                observation = obs_manager.create_observation(detections)
                obs_dict    = observation.to_dict()
            except Exception as e:
                logger.error(f"ObservationManager failed on frame {frame_count}: {e}")
                continue

            # ── Log observation structure every 15 frames ──────────────────────
            if frame_count % 15 == 0:
                print(f"\n[Frame {frame_count:04d}] --- Observation Object for Agent ---")
                try:
                    # default=str safely handles any numpy scalars that leak through
                    print(json.dumps(obs_dict, indent=4, default=str))
                except Exception as je:
                    logger.warning(f"JSON serialization failed: {je}. Raw: {obs_dict}")
                print(f"Observation Vector: {observation.to_array()}")

            # ── 4. HUD Visualization Overlay ───────────────────────────────────
            try:
                vis  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                dino = obs_dict.get("dino", {})
                obs  = obs_dict.get("obstacle", {})

                # Draw Dino — green box + velocity & jump state labels
                if isinstance(dino, dict) and dino.get("width", 0) > 0:
                    cv2.rectangle(
                        vis,
                        (dino["x"], dino["y"]),
                        (dino["x"] + dino["width"], dino["y"] + dino["height"]),
                        (0, 255, 0), 2
                    )
                    vel_str = f"Vel: {dino.get('velocity', 0.0):+.1f}"
                    cv2.putText(
                        vis, vel_str,
                        (dino["x"], max(0, dino["y"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
                    )
                    jump_lbl = "JUMPING" if dino.get("is_jumping") else "GROUND"
                    cv2.putText(
                        vis, jump_lbl,
                        (dino["x"], max(0, dino["y"] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1
                    )

                # Draw Obstacle — red (cactus) or blue (bird) box + label
                if isinstance(obs, dict) and obs.get("width", 0) > 0:
                    color = (0, 0, 255) if obs.get("type") == "cactus" else (255, 120, 0)
                    cv2.rectangle(
                        vis,
                        (obs["x"], obs["y"]),
                        (obs["x"] + obs["width"], obs["y"] + obs["height"]),
                        color, 2
                    )
                    obs_label = f"{str(obs.get('type', '')).upper()} ({obs['width']}x{obs['height']})"
                    cv2.putText(
                        vis, obs_label,
                        (obs["x"], max(0, obs["y"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                    )

                    # Distance connector line + distance label
                    if isinstance(dino, dict) and dino.get("width", 0) > 0:
                        cv2.line(
                            vis,
                            (dino["x"] + dino["width"], dino["y"] + dino["height"] // 2),
                            (obs["x"],                  obs["y"]  + obs["height"]  // 2),
                            (255, 0, 255), 2
                        )
                        mid_x = (dino["x"] + dino["width"] + obs["x"]) // 2
                        cv2.putText(
                            vis,
                            f"Dist: {obs_dict.get('distance', 0):.0f}px",
                            (mid_x - 30, max(0, dino["y"] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
                        )

                # ── HUD top bar: dark strip with all agent-facing features ─────
                hud_text = (
                    f"Dist: {obs_dict.get('distance', 0):.0f}px | "
                    f"Type: {obs_dict.get('type', 'none')} | "
                    f"Obs (W,H): ({obs_dict.get('width', 0):.0f}, {obs_dict.get('height', 0):.0f}) | "
                    f"Vel: {obs_dict.get('velocity', 0):+.1f} | "
                    f"Motion: {obs_dict.get('motion', 0):.1f}"
                )
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (30, 30, 30), -1)
                cv2.putText(vis, hud_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow(window_name, vis)

            except Exception as ve:
                logger.warning(f"Visualization error on frame {frame_count}: {ve}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        logger.info("Observation diagnostic interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in observation pipeline loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture.close()
        cv2.destroyAllWindows()
        print("[Done] Observation diagnostic finished cleanly.")

 