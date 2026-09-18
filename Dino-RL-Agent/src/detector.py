

# import cv2

# from config import DINO_ROI_WIDTH

# # defien condited selection for dino
# def inspect_dino_candidates(binary):

#     dino_region = binary[:, :DINO_ROI_WIDTH]

#     num_labels, labels, stats, centroids = (
#         cv2.connectedComponentsWithStats(
#             dino_region
#         )
#     )

#     for i in range(1, num_labels):

#         x, y, w, h, area = stats[i]

#         print(
#             f"ID={i:3d} | "
#             f"x={x:3d} | "
#             f"y={y:3d} | "
#             f"w={w:3d} | "
#             f"h={h:3d} | "
#             f"area={area:6d}"
#         )


# def detect_dino(binary):

#     dino_region = binary[:, :DINO_ROI_WIDTH]

#     num_labels, labels, stats, centroids = (
#         cv2.connectedComponentsWithStats(
#             dino_region
#         )
#     )

#     candidates = []

#     for i in range(1, num_labels):

#         x, y, w, h, area = stats[i]

#         # Ignore huge connected regions
#         if w > 80:
#             continue

#         if h > 100:
#             continue

#         if area > 5000:
#             continue

#         # Ignore tiny noise
#         if area < 20:
#             continue

#         candidates.append(
#             (x, y, w, h, area)
#         )

#     if not candidates:
#         return None

#     # Candidate closest to the expected Dino location
#     dino = min(
#         candidates,
#         key=lambda item: abs(item[0] - 50)
#     )

#     return dino


# def detect_obstacle(binary):

#     obstacle_region = binary[:, DINO_ROI_WIDTH:]

#     num_labels, labels, stats, centroids = (
#         cv2.connectedComponentsWithStats(
#             obstacle_region
#         )
#     )

#     candidates = []

#     for i in range(1, num_labels):

#         x, y, w, h, area = stats[i]

#         # Ignore tiny noise
#         if area < 20:
#             continue

#         # Ignore extremely large regions
#         if w > 100 or h > 100:
#             continue

#         # Convert ROI coordinate back to full game coordinate
#         x = x + DINO_ROI_WIDTH

#         candidates.append(
#             (x, y, w, h, area)
#         )

#     if not candidates:
#         return None

#     # Dino is on the left.
#     # The obstacle with the smallest X is the closest candidate.
#     obstacle = min(
#         candidates,
#         key=lambda item: item[0]
#     )

#     return obstacle

# --- version 2 ---
import cv2
import numpy as np


# ------------------------------------------------------------
# Game geometry
# ------------------------------------------------------------

DINO_ROI_WIDTH = 180

MIN_OBSTACLE_X = 150
MAX_OBSTACLE_X = 920

MIN_OBSTACLE_Y = 100

# Approximate ground position for your 950x300 capture.
# The detector will refine this automatically.
GROUND_Y = 216


# ============================================================
# Ground detection
# ============================================================

def detect_ground_y(binary):
    """
    Detect the horizontal ground line.

    The ground has many more foreground pixels horizontally
    than normal game objects.
    """

    height, width = binary.shape

    # Count white pixels in every row
    row_counts = np.count_nonzero(
        binary,
        axis=1
    )

    # Ground should be in lower half
    start_y = int(height * 0.55)

    # Ignore very bottom area
    end_y = int(height * 0.90)

    if end_y <= start_y:
        return GROUND_Y

    region = row_counts[start_y:end_y]

    if len(region) == 0:
        return GROUND_Y

    best_y = start_y + int(
        np.argmax(region)
    )

    # Require a reasonably long horizontal line.
    if row_counts[best_y] > width * 0.35:
        return best_y

    return GROUND_Y


# ============================================================
# Dino detection
# ============================================================

def detect_dino(binary):
    """
    Detect Dino only in the left side of the game.

    Returns:
        (x, y, w, h, area)

    or:
        None
    """

    height, width = binary.shape

    ground_y = detect_ground_y(binary)

    # --------------------------------------------------------
    # Dino ROI
    # --------------------------------------------------------

    roi_bottom = min(
        height,
        ground_y
    )

    roi = binary[
        120:roi_bottom,
        20:DINO_ROI_WIDTH
    ]

    # Small morphological cleanup
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        # Convert ROI coordinates
        x += 20
        y += 120

        # ----------------------------------------------------
        # Size filtering
        # ----------------------------------------------------

        if area < 30:
            continue

        if w < 10:
            continue

        if h < 15:
            continue

        if w > 80:
            continue

        if h > 80:
            continue

        # ----------------------------------------------------
        # Dino should be near ground
        # ----------------------------------------------------

        bottom = y + h

        if bottom < ground_y - 35:
            continue

        # ----------------------------------------------------
        # Avoid extremely wide objects
        # ----------------------------------------------------

        if w > h * 2.5:
            continue

        candidates.append(
            (x, y, w, h, area)
        )

    if not candidates:
        return None

    # Dino is closest to expected position.
    # This is much better than using only x=50.
    dino = min(
        candidates,
        key=lambda c: (
            abs((c[0] + c[2] / 2) - 80)
            +
            abs((c[1] + c[3]) - ground_y) * 0.5
        )
    )

    return dino


# ============================================================
# Obstacle detection
# ============================================================

def detect_obstacle(binary):
    """
    Detect the nearest obstacle.

    Supports:
        cactus
        bird

    Returns:
        (x, y, w, h, area)

    or:
        None
    """

    height, width = binary.shape

    ground_y = detect_ground_y(binary)

    # --------------------------------------------------------
    # Search only before the ground.
    #
    # This is the key improvement.
    # --------------------------------------------------------

    top_y = MIN_OBSTACLE_Y

    bottom_y = max(
        top_y + 10,
        ground_y - 3
    )

    roi = binary[
        top_y:bottom_y,
        MIN_OBSTACLE_X:MAX_OBSTACLE_X
    ]

    # --------------------------------------------------------
    # Close small gaps inside sprites
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(
            contour
        )

        # Convert ROI coordinates
        x += MIN_OBSTACLE_X
        y += top_y

        # ----------------------------------------------------
        # Basic noise filtering
        # ----------------------------------------------------

        if area < 20:
            continue

        if w < 5:
            continue

        if h < 5:
            continue

        if w > 100:
            continue

        if h > 100:
            continue

        # ----------------------------------------------------
        # Reject very wide horizontal noise
        # ----------------------------------------------------

        if w > h * 5:
            continue

        # ----------------------------------------------------
        # Obstacle should be reasonably below score area
        # ----------------------------------------------------

        if y < MIN_OBSTACLE_Y:
            continue

        candidates.append(
            (x, y, w, h, area)
        )

    if not candidates:
        return None

    # --------------------------------------------------------
    # Nearest obstacle to Dino
    # --------------------------------------------------------

    obstacle = min(
        candidates,
        key=lambda c: c[0]
    )

    return obstacle