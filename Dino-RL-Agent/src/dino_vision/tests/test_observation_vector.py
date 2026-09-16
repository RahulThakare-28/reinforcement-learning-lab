"""
Unit test for ObservationManager.get_observation() and relative velocity calculation.
Validates:
1. Vector structure: (distance, velocity, dino_height, dino_width, obstacle_height, obstacle_width, obstacle_type)
2. Type coding: 0 for bird, 1 for cactus, -1 for none.
3. Relative approaching velocity calculation: closure rate > 0 when obstacle approaches Dino.
4. Obstacle spawn transition: no negative velocity spike when a new obstacle appears.
5. JSON logging into Obs/ folder.
"""
import os
import sys
import json
import time
import numpy as np

# Ensure project and src are on sys.path
_here = os.path.abspath(os.path.dirname(__file__))
_src = os.path.abspath(os.path.join(_here, "..", ".."))
_root = os.path.abspath(os.path.join(_src, ".."))

for p in [_root, _src]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dino_vision.perception.observation import ObservationManager, Observation


def test_observation_vector_shape_and_types():
    print("\n--- Test 1: Observation Vector Shape & Types ---")
    obs_manager = ObservationManager()
    obs_manager.reset()

    # Simulated detection with Cactus
    cactus_detection = {
        "dino": {"x": 40, "y": 170, "width": 44, "height": 47},
        "obstacle": {"type": "cactus", "x": 300, "y": 170, "width": 25, "height": 48},
        "ground_y": 216,
    }

    vec = obs_manager.get_observation(cactus_detection)
    print(f"Cactus Observation Vector: {vec}")
    assert isinstance(vec, np.ndarray), f"Expected np.ndarray, got {type(vec)}"
    assert vec.shape == (7,), f"Expected shape (7,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected dtype float32, got {vec.dtype}"

    # Distance = 300 - (40 + 44) = 216
    assert abs(vec[0] - 216.0) < 1e-3, f"Expected distance 216, got {vec[0]}"
    # Dino height & width
    assert abs(vec[2] - 47.0) < 1e-3, f"Expected dino_height 47, got {vec[2]}"
    assert abs(vec[3] - 44.0) < 1e-3, f"Expected dino_width 44, got {vec[3]}"
    # Obstacle height & width
    assert abs(vec[4] - 48.0) < 1e-3, f"Expected obstacle_height 48, got {vec[4]}"
    assert abs(vec[5] - 25.0) < 1e-3, f"Expected obstacle_width 25, got {vec[5]}"
    # Cactus type code = 1.0
    assert abs(vec[6] - 1.0) < 1e-3, f"Expected cactus type 1.0, got {vec[6]}"
    print("  [PASS] Cactus vector shape and values verified!")

    # Simulated detection with Bird
    bird_detection = {
        "dino": {"x": 40, "y": 170, "width": 44, "height": 47},
        "obstacle": {"type": "bird", "x": 300, "y": 140, "width": 38, "height": 28},
        "ground_y": 216,
    }
    vec_bird = obs_manager.get_observation(bird_detection)
    print(f"Bird Observation Vector: {vec_bird}")
    # Bird type code = 0.0
    assert abs(vec_bird[6] - 0.0) < 1e-3, f"Expected bird type 0.0, got {vec_bird[6]}"
    print("  [PASS] Bird vector type code (0.0) verified!")


def test_relative_approaching_velocity():
    print("\n--- Test 2: Relative Approaching Velocity Calculation ---")
    obs_manager = ObservationManager(ema_alpha=0.5)
    obs_manager.reset()

    # Simulate 5 consecutive frames where the obstacle moves left by 15 pixels each frame
    # (Approaching speed = 15 px/frame)
    dino_box = {"x": 40, "y": 170, "width": 44, "height": 47}
    obs_x = 400

    print("Simulating approaching obstacle moving left at 15 px/frame:")
    for frame in range(1, 6):
        detection = {
            "dino": dino_box,
            "obstacle": {"type": "cactus", "x": obs_x, "y": 170, "width": 24, "height": 48},
            "ground_y": 216,
        }
        vec = obs_manager.get_observation(detection)
        dist = vec[0]
        v_rel = vec[1]
        print(f"  Frame {frame}: Obstacle X = {obs_x}, Distance = {dist:.1f}, Relative Velocity = {v_rel:.2f}")

        if frame >= 3:
            # Velocity should be strictly greater than 0 and approaching 15.0 px/frame
            assert v_rel > 0.0, f"Expected velocity > 0, got {v_rel}"

        obs_x -= 15  # Obstacle moves left towards Dino

    print(f"  Final Relative Velocity: {v_rel:.2f} px/frame (target: ~15.0)")
    assert v_rel > 10.0, f"Expected converged velocity near 15.0, got {v_rel}"
    print("  [PASS] Approaching relative velocity calculation successfully verified!")


def test_new_obstacle_spawn_transition():
    print("\n--- Test 3: New Obstacle Spawn Transition (No Negative Spike) ---")
    obs_manager = ObservationManager(ema_alpha=0.5)
    obs_manager.reset()

    dino_box = {"x": 40, "y": 170, "width": 44, "height": 47}

    # Frame 1 & 2: obstacle close to dino
    obs_manager.get_observation({
        "dino": dino_box,
        "obstacle": {"type": "cactus", "x": 100, "y": 170, "width": 24, "height": 48},
        "ground_y": 216,
    })
    vec_prev = obs_manager.get_observation({
        "dino": dino_box,
        "obstacle": {"type": "cactus", "x": 85, "y": 170, "width": 24, "height": 48},
        "ground_y": 216,
    })

    # Frame 3: That obstacle passed, NEW obstacle spawns at X=500
    vec_spawn = obs_manager.get_observation({
        "dino": dino_box,
        "obstacle": {"type": "bird", "x": 500, "y": 140, "width": 38, "height": 28},
        "ground_y": 216,
    })

    print(f"  Before spawn: v_rel = {vec_prev[1]:.2f}")
    print(f"  After spawn : v_rel = {vec_spawn[1]:.2f} (Distance: {vec_spawn[0]:.1f})")

    # Velocity must NOT become negative
    assert vec_spawn[1] >= 0.0, f"Velocity should not be negative on spawn, got {vec_spawn[1]}"
    print("  [PASS] Spawn transition smoothly handled without negative spike!")


def test_json_logging_to_obs_dir():
    print("\n--- Test 4: JSON Logging to Obs/ Directory ---")
    obs_manager = ObservationManager()
    obs_manager.reset()

    detection = {
        "dino": {"x": 40, "y": 170, "width": 44, "height": 47},
        "obstacle": {"type": "cactus", "x": 250, "y": 170, "width": 24, "height": 48},
        "ground_y": 216,
    }

    vec = obs_manager.get_observation(detection)
    log_file = obs_manager.log_observation(obs_manager.current_observation, frame_id=42)

    print(f"  Logged observation to: {log_file}")
    assert log_file is not None, "log_file should not be None"
    assert os.path.isfile(log_file), f"Log file does not exist: {log_file}"

    # Read last line and parse as JSON
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0, "Log file is empty"
        last_record = json.loads(lines[-1])

    print("  Sample JSON Record Content:")
    print(json.dumps(last_record, indent=2))

    assert last_record["frame_id"] == 42
    assert "observation_vector" in last_record
    assert len(last_record["observation_vector"]) == 7
    assert last_record["vector_labels"] == [
        "distance", "velocity", "dino_height", "dino_width",
        "obstacle_height", "obstacle_width", "obstacle_type"
    ]
    print("  [PASS] JSON logging to Obs/ successfully verified!")


if __name__ == "__main__":
    print("=" * 70)
    print("Running Dino Vision Observation Unit Tests")
    print("=" * 70)
    test_observation_vector_shape_and_types()
    test_relative_approaching_velocity()
    test_new_obstacle_spawn_transition()
    test_json_logging_to_obs_dir()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
