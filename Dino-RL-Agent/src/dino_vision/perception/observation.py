import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np

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
    Provides structured dictionary access, JSON serialization, and the 7-element
    numerical vector required by the reinforcement learning model.
    """

    VECTOR_LABELS = [
        "distance",
        "velocity",
        "dino_height",
        "dino_width",
        "obstacle_height",
        "obstacle_width",
        "obstacle_type",
    ]

    def __init__(
        self,
        dino: Optional[Dict[str, Any]] = None,
        obstacle: Optional[Dict[str, Any]] = None,
        distance: float = 600.0,
        velocity: float = 0.0,
        vertical_velocity: float = 0.0,
        motion: float = 0.0,
        dino_height: float = 47.0,
        dino_width: float = 44.0,
        obstacle_height: float = 0.0,
        obstacle_width: float = 0.0,
        obstacle_type: str = "none",
        obstacle_type_code: float = -1.0,
    ):
        self.dino = dino or {
            "x": 0,
            "y": 0,
            "width": int(dino_width),
            "height": int(dino_height),
            "velocity": float(vertical_velocity),
            "is_jumping": False,
        }
        self.obstacle = obstacle or {
            "type": "none",
            "x": 0,
            "y": 0,
            "width": int(obstacle_width),
            "height": int(obstacle_height),
            "distance": float(distance),
        }

        self.distance = float(distance)
        self.velocity = float(velocity)  # Relative approaching velocity v_rel
        self.vertical_velocity = float(vertical_velocity)  # Dino jumping/falling speed
        self.motion = float(motion)  # Dino y position
        self.dino_height = float(dino_height)
        self.dino_width = float(dino_width)
        self.obstacle_height = float(obstacle_height)
        self.obstacle_width = float(obstacle_width)
        self.type = str(obstacle_type).lower()
        # 0.0 for bird, 1.0 for cactus, -1.0 for none
        self.type_code = float(obstacle_type_code)

    def to_vector(self) -> np.ndarray:
        """
        Produce the canonical 7-element numerical observation vector:
        [distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type]
        where obstacle_type: 0 for bird, 1 for cactus, -1 for none.
        """
        return np.array(
            [
                self.distance,
                self.velocity,
                self.dino_height,
                self.dino_width,
                self.obstacle_height,
                self.obstacle_width,
                self.type_code,
            ],
            dtype=np.float32,
        )

    def to_array(self) -> np.ndarray:
        """Alias for to_vector() for Gym/SB3 compatibility."""
        return self.to_vector()

    def to_dict(self) -> Dict[str, Any]:
        """Produce rich dictionary representation for diagnostics, agent debugging, and logging."""
        return {
            "dino": {
                "x": int(self.dino.get("x", 0)),
                "y": int(self.dino.get("y", 0)),
                "width": int(self.dino_width),
                "height": int(self.dino_height),
                "vertical_velocity": float(self.vertical_velocity),
                "is_jumping": bool(self.dino.get("is_jumping", False)),
            },
            "obstacle": {
                "type": self.type,
                "type_code": float(self.type_code),
                "x": int(self.obstacle.get("x", 0)),
                "y": int(self.obstacle.get("y", 0)),
                "width": int(self.obstacle_width),
                "height": int(self.obstacle_height),
                "distance": float(self.distance),
            },
            "distance": float(self.distance),
            "velocity": float(self.velocity),
            "relative_velocity": float(self.velocity),
            "vertical_velocity": float(self.vertical_velocity),
            "motion": float(self.motion),
            "dino_height": float(self.dino_height),
            "dino_width": float(self.dino_width),
            "obstacle_height": float(self.obstacle_height),
            "obstacle_width": float(self.obstacle_width),
            "type": self.type,
            "type_code": float(self.type_code),
            "observation_vector": self.to_vector().tolist(),
        }

    def __repr__(self) -> str:
        return (
            f"Observation(dist={self.distance:.1f}, "
            f"v_rel={self.velocity:.1f}, "
            f"dino=({self.dino_width:.0f}x{self.dino_height:.0f}), "
            f"obs=({self.obstacle_width:.0f}x{self.obstacle_height:.0f}, {self.type}[{self.type_code:.0f}]))"
        )


class ObservationManager:
    """
    Observation Layer responsible for computing:
    - Distance: Horizontal clearance between Dino front and nearest incoming obstacle front
    - Relative Velocity (v_rel): Approaching rate of closure (delta_distance / delta_t)
    - Dino Dimensions: Height & width
    - Obstacle Dimensions: Height & width
    - Obstacle Classification: 0 for bird, 1 for cactus, -1 for none
    - Vector Formulation: get_observation() returning the exact 7-element vector
    - Data Logging: JSON/JSONL logging into the Obs/ folder for offline analysis
    """

    def __init__(
        self,
        max_distance: float = 600.0,
        ground_y_baseline: int = 216,
        ema_alpha: float = 0.5,
        obs_dir: Optional[str] = None,
    ):
        self.max_distance = float(max_distance)
        self.ground_y_baseline = ground_y_baseline
        self.ema_alpha = float(ema_alpha)

        # Resolution of Obs directory path
        if obs_dir is None:
            # Look for workspace Obs folder or project root Obs folder
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            candidate = os.path.join(base_dir, "Obs")
            if os.path.isdir(candidate) or os.path.isdir(base_dir):
                self.obs_dir = candidate
            else:
                self.obs_dir = os.path.abspath("Obs")
        else:
            self.obs_dir = os.path.abspath(obs_dir)

        # State tracking for vertical dynamics (Dino jump)
        self.previous_dino_y: Optional[float] = None

        # State tracking for horizontal dynamics (Relative velocity of approaching obstacle)
        self.previous_distance: Optional[float] = None
        self.previous_obstacle_x: Optional[float] = None
        self.previous_dino_x: Optional[float] = None
        self.previous_time: Optional[float] = None

        # Smoothed relative velocity
        self.relative_velocity: float = 0.0
        self.smoothed_velocity: float = 0.0

        # Latest cached observation object
        self.current_observation: Optional[Observation] = None

        # Session logging state
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path: Optional[str] = None

    def reset(self) -> None:
        """Reset velocity and motion tracking state for a new game / episode."""
        self.previous_dino_y = None
        self.previous_distance = None
        self.previous_obstacle_x = None
        self.previous_dino_x = None
        self.previous_time = None
        self.relative_velocity = 0.0
        self.smoothed_velocity = 0.0
        self.current_observation = None
        logger.info("ObservationManager state reset.")

    def calculate_distance(
        self, dino: Optional[Dict[str, Any]], obstacle: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate horizontal clearance between Dino's front edge and the nearest obstacle.
        Distance = obstacle_front_x - dino_front_x
        """
        try:
            if not dino or not obstacle:
                return float(self.max_distance)

            dino_front_x = float(dino.get("x", 0) + dino.get("width", 0))
            obstacle_front_x = float(obstacle.get("x", 0))

            distance = obstacle_front_x - dino_front_x
            return float(max(0.0, min(distance, self.max_distance)))
        except Exception as e:
            logger.warning(f"Error calculating distance: {e}")
            return float(self.max_distance)

    def calculate_relative_velocity(
        self,
        dino: Optional[Dict[str, Any]],
        obstacle: Optional[Dict[str, Any]],
        current_distance: float,
        current_time: Optional[float] = None,
    ) -> float:
        """
        Calculate relative approaching velocity (v_rel) between Dino and Obstacle.

        Physics Logic:
          - Dino (Object A) front edge: x_A = dino.x + dino.width
          - Obstacle (Object B) front edge: x_B = obstacle.x
          - Approaching speed: rate of closure = -(d(x_B - x_A) / dt)
          - Since Obstacle B moves LEFT towards Dino A, x_B decreases, so distance decreases:
            Delta_distance = previous_distance - current_distance > 0.
          - When an obstacle leaves and a NEW obstacle spawns far to the right:
            current_distance jumps up drastically (e.g. from 30px to 450px).
            We detect this spawn transition and preserve smoothed speed without registering
            a negative velocity spike!
          - An Exponential Moving Average (EMA) filters sub-pixel contour jitter.
        """
        try:
            if current_time is None:
                current_time = time.time()

            # If no obstacle is present on screen
            if not obstacle or obstacle.get("type", "none") == "none" or current_distance >= self.max_distance:
                # Retain previous smoothed speed or gently decay if stationary for a while
                self.previous_distance = None
                self.previous_obstacle_x = None
                return float(self.smoothed_velocity)

            obs_x = float(obstacle.get("x", 0))
            dino_front_x = float(dino.get("x", 0) + dino.get("width", 0)) if dino else 0.0

            # First frame with an obstacle detected
            if self.previous_distance is None or self.previous_obstacle_x is None:
                self.previous_distance = current_distance
                self.previous_obstacle_x = obs_x
                self.previous_dino_x = dino_front_x
                self.previous_time = current_time
                # Return current smoothed velocity if available, else initial guess
                return float(self.smoothed_velocity)

            # Check for new obstacle spawn / reset (distance jumped upward)
            dist_jump = current_distance - self.previous_distance
            obs_jump = obs_x - self.previous_obstacle_x

            if dist_jump > 40.0 or obs_jump > 40.0:
                # New obstacle entered the screen ahead! Reset baseline, do not spike
                self.previous_distance = current_distance
                self.previous_obstacle_x = obs_x
                self.previous_dino_x = dino_front_x
                self.previous_time = current_time
                return float(self.smoothed_velocity)

            # Normal consecutive frame tracking: approaching rate
            delta_distance = self.previous_distance - current_distance

            # delta_distance should be >= 0 when obstacle moves left towards Dino
            instant_velocity = max(0.0, delta_distance)

            # Update Exponential Moving Average (EMA)
            if self.smoothed_velocity == 0.0 and instant_velocity > 0:
                self.smoothed_velocity = instant_velocity
            elif instant_velocity > 0:
                self.smoothed_velocity = (self.ema_alpha * instant_velocity) + (
                    (1.0 - self.ema_alpha) * self.smoothed_velocity
                )

            # FIX 2 (Anomaly D): Safeguard velocity within realistic bounds when
            # obstacle is actively tracked. Prevents stagnant velocity (v=11.1)
            # and caps runaway estimates.
            if self.smoothed_velocity > 0.0:
                self.smoothed_velocity = float(np.clip(self.smoothed_velocity, 10.0, 45.0))

            self.relative_velocity = float(self.smoothed_velocity)

            # Update history
            self.previous_distance = current_distance
            self.previous_obstacle_x = obs_x
            self.previous_dino_x = dino_front_x
            self.previous_time = current_time

            return self.relative_velocity

        except Exception as e:
            logger.warning(f"Error calculating relative velocity: {e}")
            return float(self.smoothed_velocity)

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

            current_y = float(dino.get("y", 0.0))

            if self.previous_dino_y is None:
                vert_velocity = 0.0
            else:
                # Positive = ascending (jumping up), Negative = descending (falling down)
                vert_velocity = self.previous_dino_y - current_y

            self.previous_dino_y = current_y

            # Airborne check against baseline ground
            dino_height = float(dino.get("height", 47.0))
            is_jumping = (current_y + dino_height) < (self.ground_y_baseline - 10)

            return current_y, float(vert_velocity), is_jumping
        except Exception as e:
            logger.warning(f"Error calculating dino vertical motion: {e}")
            return 0.0, 0.0, False

    def create_observation(
        self,
        detection_result: Dict[str, Any],
        current_time: Optional[float] = None,
    ) -> Observation:
        """
        Transform detection layer output into an Observation object with full
        relative velocity, geometry, and type coding.

        Parameters
        ----------
        detection_result : dict
            Output from Detector.detect(), containing "dino", "obstacle", and "ground_y".
        current_time : float, optional
            Timestamp for velocity calculation.

        Returns
        -------
        Observation
            Structured observation object.
        """
        try:
            dino_data = detection_result.get("dino")
            obstacle_data = detection_result.get("obstacle")
            ground_y = detection_result.get("ground_y", self.ground_y_baseline)

            # 1. Dino geometry & vertical dynamics
            dino_height = float(dino_data.get("height", 47.0)) if dino_data else 47.0
            dino_width = float(dino_data.get("width", 44.0)) if dino_data else 44.0
            motion, vert_velocity, is_jumping = self.calculate_dino_motion(dino_data)

            # 2. Horizontal Distance
            distance = self.calculate_distance(dino_data, obstacle_data)

            # 3. Relative Velocity (rate of closure between Dino and obstacle)
            rel_velocity = self.calculate_relative_velocity(
                dino_data, obstacle_data, distance, current_time=current_time
            )

            # Enrich dino dictionary
            enriched_dino = None
            if dino_data:
                enriched_dino = {
                    "x": int(dino_data.get("x", 0)),
                    "y": int(dino_data.get("y", 0)),
                    "width": int(dino_width),
                    "height": int(dino_height),
                    "vertical_velocity": float(vert_velocity),
                    "is_jumping": bool(is_jumping),
                }

            # FIX 2 (Anomaly C): Clamp Dino dimensions to anatomically valid range.
            # Prevents out-of-distribution vectors from contour fragmentation.
            dino_width = float(np.clip(dino_width, 40.0, 75.0))
            dino_height = float(np.clip(dino_height, 30.0, 65.0))

            # 4. Obstacle Geometry & Type Code
            # Required by specification:
            # 0 for bird
            # 1 for cactus
            # -1 for none / no obstacle
            if obstacle_data and obstacle_data.get("type", "none") != "none":
                obs_type = str(obstacle_data.get("type", "cactus")).lower()
                obs_width = float(obstacle_data.get("width", 0.0))
                obs_height = float(obstacle_data.get("height", 0.0))

                if "bird" in obs_type:
                    type_code = 0.0
                    obs_type = "bird"
                    # FIX 2: Clamp bird dimensions
                    obs_width = float(np.clip(obs_width, 15.0, 100.0))
                    obs_height = float(np.clip(obs_height, 12.0, 40.0))
                else:
                    type_code = 1.0
                    obs_type = "cactus"
                    # FIX 2: Clamp cactus dimensions (Anomaly A fix at observation layer)
                    obs_width = float(np.clip(obs_width, 15.0, 100.0))
                    obs_height = float(np.clip(obs_height, 12.0, 75.0))

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
                type_code = -1.0
                enriched_obstacle = {
                    "type": "none",
                    "x": 0,
                    "y": 0,
                    "width": 0,
                    "height": 0,
                    "distance": float(distance),
                }

            obs = Observation(
                dino=enriched_dino,
                obstacle=enriched_obstacle,
                distance=distance,
                velocity=rel_velocity,
                vertical_velocity=vert_velocity,
                motion=motion,
                dino_height=dino_height,
                dino_width=dino_width,
                obstacle_height=obs_height,
                obstacle_width=obs_width,
                obstacle_type=obs_type,
                obstacle_type_code=type_code,
            )

            self.current_observation = obs
            return obs

        except Exception as e:
            logger.error(f"Error creating observation: {e}")
            fallback = Observation(
                distance=self.max_distance,
                velocity=0.0,
                vertical_velocity=0.0,
                motion=0.0,
                dino_height=47.0,
                dino_width=44.0,
                obstacle_height=0.0,
                obstacle_width=0.0,
                obstacle_type="none",
                obstacle_type_code=-1.0,
            )
            self.current_observation = fallback
            return fallback

    def get_observation(
        self,
        detection_result: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Produce the numerical observation vector with:
        (distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type)

        In vector:
          - 0 for bird
          - 1 for cactus
          - (-1 for none)

        Parameters
        ----------
        detection_result : dict, optional
            Detection dictionary from detector.detect(). If provided, updates observation state.
            If None, returns the vector from the latest computed observation.
        current_time : float, optional
            Timestamp used for velocity calculation.

        Returns
        -------
        np.ndarray
            1D float32 array of shape (7,):
            [distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type]
        """
        if detection_result is not None:
            obs = self.create_observation(detection_result, current_time=current_time)
            return obs.to_vector()

        if self.current_observation is not None:
            return self.current_observation.to_vector()

        # Default fallback vector if called before any detection
        return np.array(
            [self.max_distance, 0.0, 47.0, 44.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        )

    def log_observation(
        self,
        observation: Optional[Union[Observation, np.ndarray, Dict[str, Any]]] = None,
        frame_id: Optional[int] = None,
        session_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Log capture and observation data into JSON Lines (.jsonl) format inside the Obs/ directory.

        Parameters
        ----------
        observation : Observation, np.ndarray, or dict, optional
            Observation to log. If None, uses self.current_observation.
        frame_id : int, optional
            Current frame number.
        session_id : str, optional
            Custom session ID for log filename. Defaults to self.session_id.
        extra : dict, optional
            Extra metadata to include in the record.

        Returns
        -------
        str or None
            Path to the written log file.
        """
        try:
            # Ensure target Obs directory exists
            os.makedirs(self.obs_dir, exist_ok=True)

            if session_id is None:
                session_id = self.session_id

            if self.log_file_path is None or session_id != self.session_id:
                self.session_id = session_id
                self.log_file_path = os.path.join(self.obs_dir, f"obs_session_{session_id}.jsonl")

            # Determine observation details
            if observation is None:
                observation = self.current_observation

            if isinstance(observation, Observation):
                obs_dict = observation.to_dict()
                obs_vec = observation.to_vector().tolist()
            elif isinstance(observation, np.ndarray):
                obs_vec = observation.tolist()
                obs_dict = {
                    label: val
                    for label, val in zip(Observation.VECTOR_LABELS, obs_vec)
                }
            elif isinstance(observation, dict):
                obs_dict = observation
                obs_vec = observation.get("observation_vector", [])
            else:
                obs_vec = self.get_observation().tolist()
                obs_dict = {
                    label: val
                    for label, val in zip(Observation.VECTOR_LABELS, obs_vec)
                }

            record = {
                "frame_id": frame_id,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "observation_vector": obs_vec,
                "vector_labels": Observation.VECTOR_LABELS,
                "details": obs_dict,
            }

            if extra:
                record["extra"] = extra

            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

            return self.log_file_path

        except Exception as e:
            logger.error(f"Failed to log observation to JSON: {e}")
            return None


# =====================================================================
# Interactive Diagnostic & Testing
# =====================================================================
if __name__ == "__main__":
    import sys
    import cv2

    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from dino_vision.capture.screen_capture import ScreenCapture
    from dino_vision.preprocessing.preprocessor import Preprocessor
    from dino_vision.detection.detector import Detector

    print("=" * 75)
    print("Dino Vision: ScreenCapture -> Preprocessor -> Detector -> Observation")
    print("=" * 75)
    print("Pipeline Flow:")
    print("  [ScreenCapture] -> RGB Frame")
    print("  -> [Preprocessor] -> 1-Channel Binary")
    print("  -> [Detector]     -> {dino, obstacle, ground_y}")
    print("  -> [Observation]  -> get_observation() [7-Element Vector]")
    print()
    print("Vector Format: (distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type)")
    print("Type Coding  : 0 = Bird, 1 = Cactus, -1 = None")
    print()
    print("Controls (OpenCV window focused):")
    print("  'q' / ESC — quit cleanly")
    print("=" * 75)

    preprocessor = Preprocessor(auto_invert=True)
    detector = Detector()
    obs_manager = ObservationManager()
    capture = ScreenCapture()
    window_name = "Dino Vision - Agent Observation HUD"

    try:
        capture.select_region()
        capture.play()

        frame_count = 0
        print(f"[INFO] Streaming active. Logs saving to: {obs_manager.obs_dir}\n")

        for frame in capture.start(fps_limit=30):
            frame_count += 1
            now = time.time()

            # ── 1. Preprocess ──────────────────────────────────────────────────
            try:
                prep_data = preprocessor.process(frame)
                binary = prep_data["binary"]
            except Exception as e:
                logger.error(f"Preprocessor failed on frame {frame_count}: {e}")
                continue

            # ── 2. Detect ──────────────────────────────────────────────────────
            try:
                detections = detector.detect(binary)
                if not isinstance(detections, dict):
                    logger.error(f"Detector returned unexpected type on frame {frame_count}. Skipping.")
                    continue
            except Exception as e:
                logger.error(f"Detector failed on frame {frame_count}: {e}")
                continue

            # ── 3. Formulate Observation Vector ────────────────────────────────
            try:
                # Call get_observation() as requested
                obs_vector = obs_manager.get_observation(detections, current_time=now)
                observation = obs_manager.current_observation
                obs_dict = observation.to_dict() if observation else {}
            except Exception as e:
                logger.error(f"ObservationManager failed on frame {frame_count}: {e}")
                continue

            # ── 4. Print Observation Vector on Terminal & Log to Obs/ ──────────
            if frame_count % 15 == 0:
                print(
                    f"[Frame {frame_count:04d}] Observation Vector: {obs_vector.round(2).tolist()}"
                )
                print(
                    f"             Labels: (dist={obs_vector[0]:.1f}, vel={obs_vector[1]:.1f}, "
                    f"dino_h={obs_vector[2]:.0f}, dino_w={obs_vector[3]:.0f}, "
                    f"obs_h={obs_vector[4]:.0f}, obs_w={obs_vector[5]:.0f}, type={obs_vector[6]:.0f})"
                )

            # Log captured data into JSON format in Obs/ folder
            obs_manager.log_observation(observation, frame_id=frame_count)

            # ── 5. HUD Visualization Overlay ───────────────────────────────────
            try:
                vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                dino = obs_dict.get("dino", {})
                obs = obs_dict.get("obstacle", {})

                # Draw Dino — green box + velocity & jump state labels
                if isinstance(dino, dict) and dino.get("width", 0) > 0:
                    cv2.rectangle(
                        vis,
                        (dino["x"], dino["y"]),
                        (dino["x"] + dino["width"], dino["y"] + dino["height"]),
                        (0, 255, 0), 2,
                    )
                    jump_lbl = "JUMPING" if dino.get("is_jumping") else "GROUND"
                    cv2.putText(
                        vis, jump_lbl,
                        (dino["x"], max(0, dino["y"] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1,
                    )

                # Draw Obstacle — red (cactus) or blue (bird) box + label
                if isinstance(obs, dict) and obs.get("width", 0) > 0:
                    color = (0, 0, 255) if obs.get("type") == "cactus" else (255, 120, 0)
                    cv2.rectangle(
                        vis,
                        (obs["x"], obs["y"]),
                        (obs["x"] + obs["width"], obs["y"] + obs["height"]),
                        color, 2,
                    )
                    obs_label = f"{str(obs.get('type', '')).upper()} (code={obs_vector[6]:.0f})"
                    cv2.putText(
                        vis, obs_label,
                        (obs["x"], max(0, obs["y"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    )

                    # Connector distance line
                    if isinstance(dino, dict) and dino.get("width", 0) > 0:
                        cv2.line(
                            vis,
                            (dino["x"] + dino["width"], dino["y"] + dino["height"] // 2),
                            (obs["x"], obs["y"] + obs["height"] // 2),
                            (255, 0, 255), 2,
                        )
                        mid_x = (dino["x"] + dino["width"] + obs["x"]) // 2
                        cv2.putText(
                            vis,
                            f"Dist: {obs_vector[0]:.0f}px | V_rel: {obs_vector[1]:.1f}",
                            (max(10, mid_x - 60), max(20, dino["y"] - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                        )

                # ── HUD top bar: all 7 vector elements ─────────────────────────
                hud_text = (
                    f"Dist:{obs_vector[0]:.0f} | "
                    f"Vel:{obs_vector[1]:.1f} | "
                    f"Dino:({obs_vector[3]:.0f}x{obs_vector[2]:.0f}) | "
                    f"Obs:({obs_vector[5]:.0f}x{obs_vector[4]:.0f}) | "
                    f"Type:{obs_dict.get('type', 'none')}({obs_vector[6]:.0f})"
                )
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (30, 30, 30), -1)
                cv2.putText(
                    vis, hud_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1
                )

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