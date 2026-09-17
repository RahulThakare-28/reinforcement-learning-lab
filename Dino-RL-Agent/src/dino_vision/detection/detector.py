import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("DinoVision.Detector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [Detector] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Detector:
    """
    Object detection layer for Chrome Dino.
    Extracts the Dino character and the nearest incoming obstacle (cactus or bird)
    from a 1-channel binary frame.
    """

    def __init__(
        self,
        dino_roi_width: int = 180,
        min_obstacle_x: int = 140,
        ground_y_default: int = 216,
        min_area: int = 15,
    ):
        """
        Parameters
        ----------
        dino_roi_width : int
            Horizontal width on the left side of the screen reserved for finding the Dino.
        min_obstacle_x : int
            Minimum X coordinate to search for obstacles ahead of Dino.
        ground_y_default : int
            Fallback ground line Y coordinate if auto-detection fails.
        min_area : int
            Minimum contour pixel area to filter out tiny noise.
        """
        self.dino_roi_width = dino_roi_width
        self.min_obstacle_x = min_obstacle_x
        self.ground_y_default = ground_y_default
        self.min_area = min_area
        self.last_dino: Optional[Dict[str, int]] = None
        self.last_ground_y: int = ground_y_default

    def detect_ground_y(self, binary_mask: np.ndarray) -> int:
        """
        Detect the horizontal ground line Y-coordinate.
        Expects mask where foreground objects/ground are non-zero (255).
        """
        try:
            height, width = binary_mask.shape
            start_y = int(height * 0.50)
            end_y = int(height * 0.95)

            if end_y <= start_y:
                return self.last_ground_y

            row_counts = np.count_nonzero(binary_mask[start_y:end_y, :], axis=1)
            if len(row_counts) == 0:
                return self.last_ground_y

            best_offset = int(np.argmax(row_counts))
            best_y = start_y + best_offset

            # Require a solid horizontal cluster of pixels to consider it ground
            if row_counts[best_offset] > (width * 0.15):
                self.last_ground_y = best_y
                return best_y

            return self.last_ground_y
        except Exception as e:
            logger.warning(f"Error detecting ground Y: {e}")
            return self.last_ground_y

    def _prepare_mask(self, binary_frame: np.ndarray) -> np.ndarray:
        """
        Ensure mask has foreground objects as WHITE (255) and background as BLACK (0)
        for OpenCV contour detection.
        """
        if binary_frame is None:
            raise ValueError("Input binary_frame is None.")

        if len(binary_frame.shape) != 2:
            raise ValueError(f"Detector expects a single-channel 2D image, got shape {binary_frame.shape}")

        # If background is white (mean > 128) and objects are black (0), invert it
        if binary_frame.mean() > 128:
            return cv2.bitwise_not(binary_frame)
        return binary_frame.copy()

    def detect_dino(self, mask: np.ndarray, ground_y: int) -> Optional[Dict[str, int]]:
        """
        Locate the Dino character in the left ROI of the screen.
        """
        try:
            height, width = mask.shape
            roi_right = min(width, self.dino_roi_width)
            roi_bottom = min(height, ground_y + 10)

            # Slices left region where Dino lives
            roi = mask[40:roi_bottom, 10:roi_right]
            if roi.size == 0:
                return self.last_dino

            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                # Convert local ROI coordinates back to full screen frame coordinates
                x += 10
                y += 40

                # Filter out tiny noise and ground fragments
                if area < self.min_area or w < 15 or h < 20:
                    continue

                # Filter out extremely wide ground lines
                if w > 120 or (w > h * 2.5 and h < 25):
                    continue

                candidates.append({"x": x, "y": y, "width": w, "height": h, "area": area})

            if not candidates:
                return self.last_dino

            # Merge vertically adjacent/overlapping Dino fragments (e.g. legs + body)
            if len(candidates) > 1:
                # Group candidates that are horizontally aligned (within 20px)
                aligned = [c for c in candidates if abs(c["x"] - candidates[0]["x"]) < 25]
                if len(aligned) > 1:
                    min_x = min(c["x"] for c in aligned)
                    max_x = max(c["x"] + c["width"] for c in aligned)
                    min_y = min(c["y"] for c in aligned)
                    max_y = max(c["y"] + c["height"] for c in aligned)
                    merged_w = max_x - min_x
                    merged_h = max_y - min_y
                    if 35 <= merged_w <= 85 and 30 <= merged_h <= 65:
                        candidates.append({
                            "x": min_x,
                            "y": min_y,
                            "width": merged_w,
                            "height": merged_h,
                            "area": sum(c["area"] for c in aligned),
                        })

            # Dino is typically around x ≈ 40-90. Select candidate closest to standard Dino position
            best_dino = min(
                candidates,
                key=lambda c: abs((c["x"] + c["width"] / 2) - 75) + abs((c["y"] + c["height"]) - ground_y) * 0.4
            )

            # Anatomical sanity check: Dino width is [35..75], height is [30..65]
            cand_w = int(best_dino["width"])
            cand_h = int(best_dino["height"])
            if cand_w < 30 or cand_h < 25:
                if self.last_dino is not None:
                    return self.last_dino

            dino_result = {
                "x": int(best_dino["x"]),
                "y": int(best_dino["y"]),
                "width": int(np.clip(cand_w, 35, 75)),
                "height": int(np.clip(cand_h, 30, 65)),
            }
            self.last_dino = dino_result
            return dino_result

        except Exception as e:
            logger.error(f"Error during Dino detection: {e}")
            return self.last_dino

    def detect_obstacle(
        self, mask: np.ndarray, ground_y: int, dino_x_end: int = 120
    ) -> Optional[Dict[str, Any]]:
        """
        Detect the nearest upcoming obstacle (cactus or bird) ahead of the Dino.
        Strictly excludes ground line to prevent ground-obstacle merging.
        """
        try:
            height, width = mask.shape
            search_start_x = max(dino_x_end + 5, self.min_obstacle_x)
            search_end_x = width - 5

            # Restrict search area to active track zone above ground line
            top_y = max(40, ground_y - 130)
            # FIX 1: Cut off strictly ABOVE ground line (ground_y - 2) so ground texture is never included
            bottom_y = min(height, ground_y - 2)

            if search_start_x >= search_end_x or top_y >= bottom_y:
                return None

            roi = mask[top_y:bottom_y, search_start_x:search_end_x].copy()
            if roi.size == 0:
                return None

            # Blank out any residual ground pixels near the bottom edge
            ground_cutoff = max(0, (ground_y - 2) - top_y)
            if ground_cutoff < roi.shape[0]:
                roi[ground_cutoff:, :] = 0

            # Morphological close to merge multi-branch cacti segments cleanly without ground bridging
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                closed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                # Convert to full screen coordinates
                x += search_start_x
                y += top_y

                # Noise & size filtering (minimum dimensions)
                if area < self.min_area or w < 8 or h < 10:
                    continue

                # FIX 1 & 2: Filter out ground line fragments and abnormally wide contours
                # Single/triple cactus clusters in Chrome Dino are at most ~85-110px wide.
                # A contour with w > 130px is an artifact or merged ground line!
                if w > 130:
                    continue

                # Filter out flat horizontal lines (ground bumps/debris)
                if w > h * 3.2 and h < 20:
                    continue

                # Filter out restart button or game over text (usually centered high)
                if abs(x - width / 2) < 45 and y < ground_y - 70 and w > 35:
                    continue

                # Classify obstacle type: Cactus vs Bird
                # Since ROI is cut at ground_y - 2, a cactus resting on ground will have
                # its bottom edge within 10px of ground_y. Airborne birds fly higher.
                obstacle_bottom = y + h
                if obstacle_bottom >= (ground_y - 12):
                    obstacle_type = "cactus"
                    # Restore true ground-anchored height since bottom 2px were cropped
                    h_adjusted = min(75, h + 2)
                else:
                    obstacle_type = "bird"
                    h_adjusted = h

                # Clamp dimensions to realistic physical limits
                clamped_w = int(np.clip(w, 15, 100))
                clamped_h = int(np.clip(h_adjusted, 12, 75))

                candidates.append({
                    "type": obstacle_type,
                    "x": int(x),
                    "y": int(y),
                    "width": clamped_w,
                    "height": clamped_h,
                    "area": float(area),
                })

            if not candidates:
                return None

            # The nearest obstacle to Dino is the candidate with the smallest X coordinate
            nearest = min(candidates, key=lambda o: o["x"])
            return {
                "type": nearest["type"],
                "x": nearest["x"],
                "y": nearest["y"],
                "width": nearest["width"],
                "height": nearest["height"],
            }

        except Exception as e:
            logger.error(f"Error during obstacle detection: {e}")
            return None

    def detect(self, binary_frame: np.ndarray) -> Dict[str, Any]:
        """
        Execute full detection on a binary frame.

        Parameters
        ----------
        binary_frame : np.ndarray
            Single-channel binary image (either white or black background).

        Returns
        -------
        dict
            Structured detection object for agent:
            {
                "dino": {
                    "x": int,
                    "y": int,
                    "width": int,
                    "height": int
                } or None,
                "obstacle": {
                    "type": "cactus" | "bird" | "none",
                    "x": int,
                    "y": int,
                    "width": int,
                    "height": int
                } or None,
                "ground_y": int
            }
        """
        try:
            mask = self._prepare_mask(binary_frame)
            ground_y = self.detect_ground_y(mask)

            # 1. Detect Dino
            dino = self.detect_dino(mask, ground_y)
            dino_x_end = (dino["x"] + dino["width"]) if dino else self.dino_roi_width

            # 2. Detect Nearest Obstacle ahead of Dino
            obstacle = self.detect_obstacle(mask, ground_y, dino_x_end=dino_x_end)

            return {
                "dino": dino,
                "obstacle": obstacle,
                "ground_y": ground_y,
            }

        except Exception as e:
            logger.error(f"Unexpected error in detect(): {e}")
            return {
                "dino": self.last_dino,
                "obstacle": None,
                "ground_y": self.last_ground_y,
            }


# =====================================================================
# Interactive Diagnostic & Testing
# =====================================================================
if __name__ == "__main__":
    import sys
    import os
    import json
    import time

    # Setup path
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from dino_vision.capture.screen_capture import ScreenCapture
    from dino_vision.preprocessing.preprocessor import Preprocessor

    print("=" * 70)
    print("Dino Vision: Detection Layer Diagnostic")
    print("=" * 70)

    preprocessor = Preprocessor(auto_invert=True)
    detector = Detector()
    capture = ScreenCapture()
    window_name = "Dino Vision - Detection Layer Overlay"

    try:
        capture.select_region()
        capture.play()

        print("[INFO] Running Detection Layer Live Test. Press 'q' to exit.\n")
        frame_idx = 0

        for frame in capture.start(fps_limit=30):
            frame_idx += 1

            # ── Per-frame try/except: one bad frame must never crash the loop ──
            try:
                data   = preprocessor.process(frame)
                binary = data["binary"]
            except Exception as e:
                logger.error(f"Preprocessor failed on frame {frame_idx}: {e}")
                continue

            try:
                result = detector.detect(binary)
            except Exception as e:
                logger.error(f"Detector failed on frame {frame_idx}: {e}")
                continue

            # Guard: detect() must return a dict — never a list or other type
            if not isinstance(result, dict):
                logger.error(
                    f"detector.detect() returned unexpected type "
                    f"'{type(result).__name__}' on frame {frame_idx}. "
                    "Expected dict. Skipping frame. "
                    "(If this persists, reload the module / restart the kernel.)"
                )
                continue

            # ── Log detection structure every 15 frames ───────────────────────
            if frame_idx % 15 == 0:
                print(f"\n--- Frame {frame_idx:04d} | Detection Structure ---")
                try:
                    # default=str safely converts any numpy scalars that slip through
                    print(json.dumps(result, indent=4, default=str))
                except Exception as je:
                    logger.warning(f"JSON serialization failed: {je}. Raw result: {result}")

            # ── Visualization overlay ─────────────────────────────────────────
            try:
                vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

                # Draw Dino box — green
                dino = result.get("dino")
                if isinstance(dino, dict) and dino.get("width", 0) > 0:
                    cv2.rectangle(
                        vis,
                        (dino["x"], dino["y"]),
                        (dino["x"] + dino["width"], dino["y"] + dino["height"]),
                        (0, 200, 0), 2
                    )
                    cv2.putText(
                        vis, "Dino",
                        (dino["x"], max(0, dino["y"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1
                    )

                # Draw Obstacle box — red for cactus, blue for bird
                obs = result.get("obstacle")
                if isinstance(obs, dict) and obs.get("width", 0) > 0:
                    color = (0, 0, 255) if obs.get("type") == "cactus" else (255, 100, 0)
                    cv2.rectangle(
                        vis,
                        (obs["x"], obs["y"]),
                        (obs["x"] + obs["width"], obs["y"] + obs["height"]),
                        color, 2
                    )
                    obs_label = f"{str(obs.get('type', '')).upper()} ({obs['width']}x{obs['height']})"
                    cv2.putText(
                        vis, obs_label,
                        (obs["x"], max(0, obs["y"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                    )

                    # Distance connector line between Dino and Obstacle
                    if isinstance(dino, dict) and dino.get("width", 0) > 0:
                        dist = max(0, obs["x"] - (dino["x"] + dino["width"]))
                        cv2.line(
                            vis,
                            (dino["x"] + dino["width"], dino["y"] + dino["height"] // 2),
                            (obs["x"],                  obs["y"]  + obs["height"]  // 2),
                            (255, 0, 255), 1
                        )
                        cv2.putText(
                            vis, f"Dist: {dist}px",
                            (dino["x"] + dino["width"] + 10, dino["y"] + dino["height"] // 2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                        )

                # Ground line — grey
                ground_y = result.get("ground_y", 0)
                if ground_y > 0:
                    cv2.line(vis, (0, ground_y), (vis.shape[1], ground_y), (128, 128, 128), 1)

                cv2.imshow(window_name, vis)

            except Exception as ve:
                logger.warning(f"Visualization error on frame {frame_idx}: {ve}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        logger.info("Detection diagnostic interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in detection loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture.close()
        cv2.destroyAllWindows()
        print("[Done] Detection diagnostic finished cleanly.")