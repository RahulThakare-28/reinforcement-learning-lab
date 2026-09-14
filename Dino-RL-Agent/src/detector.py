

import cv2

from config import DINO_ROI_WIDTH

# defien condited selection for dino
def inspect_dino_candidates(binary):

    dino_region = binary[:, :DINO_ROI_WIDTH]

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            dino_region
        )
    )

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]

        print(
            f"ID={i:3d} | "
            f"x={x:3d} | "
            f"y={y:3d} | "
            f"w={w:3d} | "
            f"h={h:3d} | "
            f"area={area:6d}"
        )


def detect_dino(binary):

    dino_region = binary[:, :DINO_ROI_WIDTH]

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            dino_region
        )
    )

    candidates = []

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]

        # Ignore huge connected regions
        if w > 80:
            continue

        if h > 100:
            continue

        if area > 5000:
            continue

        # Ignore tiny noise
        if area < 20:
            continue

        candidates.append(
            (x, y, w, h, area)
        )

    if not candidates:
        return None

    # Candidate closest to the expected Dino location
    dino = min(
        candidates,
        key=lambda item: abs(item[0] - 50)
    )

    return dino


def detect_obstacle(binary):

    obstacle_region = binary[:, DINO_ROI_WIDTH:]

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            obstacle_region
        )
    )

    candidates = []

    for i in range(1, num_labels):

        x, y, w, h, area = stats[i]

        # Ignore tiny noise
        if area < 20:
            continue

        # Ignore extremely large regions
        if w > 100 or h > 100:
            continue

        # Convert ROI coordinate back to full game coordinate
        x = x + DINO_ROI_WIDTH

        candidates.append(
            (x, y, w, h, area)
        )

    if not candidates:
        return None

    # Dino is on the left.
    # The obstacle with the smallest X is the closest candidate.
    obstacle = min(
        candidates,
        key=lambda item: item[0]
    )

    return obstacle