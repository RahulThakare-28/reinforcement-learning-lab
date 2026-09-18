
# import numpy as np

# from screen import capture_game, preprocess_frame
# from detector import detect_dino, detect_obstacle

# previous_dino_y = None

# def calculate_distance(dino, obstacle):

#     if dino is None or obstacle is None:
#         return 500.0

#     dino_x, dino_y, dino_w, dino_h, _ = dino

#     obstacle_x, obstacle_y, obstacle_w, obstacle_h, _ = obstacle

#     dino_right = dino_x + dino_w

#     distance = obstacle_x - dino_right

#     return max(0.0, float(distance))



# def detect_obstacle_type(obstacle):

#     if obstacle is None:
#         return 0

#     x, y, w, h, area = obstacle

#     # Temporary threshold.
#     # We will calibrate this from actual Chrome screenshots.
#     if y < 100:
#         return 1       # air

#     return 0           # ground

# def detect_obstacle_height(obstacle):

#     if obstacle is None:
#         return 0.0

#     return float(obstacle[3])


# def calculate_dino_motion(dino):

#     global previous_dino_y

#     if dino is None:
#         return 0.0, 0.0

#     x, y, w, h, area = dino

#     current_y = float(y)

#     if previous_dino_y is None:
#         velocity = 0.0
#     else:
#         velocity = previous_dino_y - current_y

#     previous_dino_y = current_y

#     return current_y, velocity



# def get_observation(frame):

#     img, gray, binary = preprocess_frame(frame)

#     dino = detect_dino(binary)

#     obstacle = detect_obstacle(binary)

#     distance = calculate_distance(
#         dino,
#         obstacle
#     )

#     obstacle_type = detect_obstacle_type(
#         obstacle
#     )

#     obstacle_height = detect_obstacle_height(
#         obstacle
#     )

#     dino_height, dino_velocity = (
#         calculate_dino_motion(dino)
#     )

#     observation = np.array(
#         [
#             distance,
#             obstacle_type,
#             obstacle_height,
#             dino_height,
#             dino_velocity
#         ],
#         dtype=np.float32
#     )

#     return observation

# -- verion 2 ---
import numpy as np

from screen import capture_game, preprocess_frame
from detector import detect_dino, detect_obstacle


previous_dino_y = None


# ============================================================
# Distance
# ============================================================

def calculate_distance(dino, obstacle):

    if dino is None or obstacle is None:
        return 500.0

    dino_x, dino_y, dino_w, dino_h, _ = dino

    obstacle_x, obstacle_y, obstacle_w, obstacle_h, _ = obstacle

    dino_right = dino_x + dino_w

    distance = obstacle_x - dino_right

    return max(
        0.0,
        float(distance)
    )


# ============================================================
# Obstacle type
# ============================================================

def detect_obstacle_type(obstacle):

    if obstacle is None:
        return 0

    x, y, w, h, area = obstacle

    # --------------------------------------------------------
    # Chrome Dino ground level is around y = 216 in the
    # current 950x300 capture.
    #
    # A cactus reaches close to the ground.
    # A bird is above the ground.
    # --------------------------------------------------------

    GROUND_Y = 216

    bottom = y + h

    ground_threshold = GROUND_Y - 12

    if bottom >= ground_threshold:
        return 0       # ground / cactus

    return 1           # air / bird


# ============================================================
# Obstacle height
# ============================================================

def detect_obstacle_height(obstacle):

    if obstacle is None:
        return 0.0

    return float(
        obstacle[3]
    )


# ============================================================
# Dino motion
# ============================================================

def calculate_dino_motion(dino):

    global previous_dino_y

    if dino is None:

        return 0.0, 0.0

    x, y, w, h, area = dino

    current_y = float(y)

    if previous_dino_y is None:

        velocity = 0.0

    else:

        velocity = (
            previous_dino_y
            -
            current_y
        )

    previous_dino_y = current_y

    return current_y, velocity


# ============================================================
# RL Observation
# ============================================================

def get_observation(frame):

    img, gray, binary = preprocess_frame(
        frame
    )

    dino = detect_dino(
        binary
    )

    obstacle = detect_obstacle(
        binary
    )

    distance = calculate_distance(
        dino,
        obstacle
    )

    obstacle_type = detect_obstacle_type(
        obstacle
    )

    obstacle_height = detect_obstacle_height(
        obstacle
    )

    dino_height, dino_velocity = (
        calculate_dino_motion(
            dino
        )
    )

    # --------------------------------------------------------
    # DO NOT CHANGE THIS.
    #
    # RL model expects exactly 5 observations.
    # --------------------------------------------------------

    observation = np.array(
        [
            distance,
            obstacle_type,
            obstacle_height,
            dino_height,
            dino_velocity
        ],
        dtype=np.float32
    )

    return observation