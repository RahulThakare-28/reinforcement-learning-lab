import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Generator

import tkinter as tk
import pyautogui
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Configure structured logging with timestamp and component tag
logger = logging.getLogger("ScreenCapture")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [ScreenCapture] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VideoRecorder:
    """
    Utility class to record streaming frames into a video file using OpenCV.
    """

    def __init__(self, output_path: str, fps: float = 30.0, codec: str = "mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.writer: Optional[Any] = None
        self.frame_size: Optional[tuple] = None
        self.is_recording = False
        self.frames_written = 0

    def start(self, width: int, height: int) -> bool:
        """Initialize the OpenCV VideoWriter."""
        if cv2 is None:
            logger.error("OpenCV (cv2) is required for video recording but is not installed.")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                logger.error(f"Failed to open video writer for path: {self.output_path}")
                return False

            self.frame_size = (width, height)
            self.is_recording = True
            self.frames_written = 0
            logger.info(f"Started video recording: {self.output_path} ({width}x{height} @ {self.fps} FPS)")
            return True
        except Exception as e:
            logger.error(f"Exception while initializing VideoWriter: {e}")
            self.is_recording = False
            return False

    def write_frame(self, frame_rgb: np.ndarray) -> bool:
        """Write a single RGB frame into the video file."""
        if not self.is_recording or self.writer is None or cv2 is None:
            return False

        try:
            h, w = frame_rgb.shape[:2]
            if self.frame_size is not None and (w != self.frame_size[0] or h != self.frame_size[1]):
                frame_rgb = cv2.resize(frame_rgb, self.frame_size)

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)
            self.frames_written += 1
            return True
        except Exception as e:
            logger.warning(f"Error writing frame to video: {e}")
            return False

    def stop(self) -> str:
        """Release the VideoWriter and save the video file."""
        if self.writer is not None:
            try:
                self.writer.release()
                logger.info(f"Video saved successfully: {self.output_path} (Total frames: {self.frames_written})")
            except Exception as e:
                logger.error(f"Error closing VideoWriter: {e}")
            finally:
                self.writer = None
        self.is_recording = False
        return self.output_path


class ScreenCapture:
    """
    Capture the screen area currently occupied by a movable, resizable capture window.

    Key Features:
    - Standard OS window controls (Minimize, Maximize, Close)
    - On-window control toolbar: Play (Start Feed), Pause (Stop Feed), Cancel (Close Window)
    - Real-time region coordinate feedback
    - Clean shutdown handling that prevents application hangs and crashes
    - Robust try-except error handling with detailed terminal logging
    - Video recording and snapshot export utilities
    """

    def __init__(
        self,
        default_geometry: str = "800x350+300+200",
        window_alpha: float = 1.0,
        transparent_color: str = "#000001"
    ):
        self.default_geometry = default_geometry
        self.window_alpha = window_alpha
        self.transparent_color = transparent_color

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[tk.Label] = None
        self.region_label: Optional[tk.Label] = None
        self.play_btn: Optional[tk.Button] = None
        self.pause_btn: Optional[tk.Button] = None

        self.running: bool = False
        self.is_paused: bool = False
        self._is_alive: bool = False
        self._capture_error_logged: bool = False

        self.video_recorder: Optional[VideoRecorder] = None
        self.last_region: Dict[str, int] = {"left": 0, "top": 0, "width": 0, "height": 0}

    def select_region(self) -> None:
        """
        Create and display the movable/resizable capture window with control buttons.
        """
        if self.is_active():
            logger.info("Capture window already open.")
            return

        try:
            self.root = tk.Tk()
            self.root.title("Dino Vision - Capture Area")
            self.root.geometry(self.default_geometry)
            self.root.attributes("-topmost", True)

            # Make the canvas area inside the red border 100% transparent on Windows
            # View ground is crystal clear while toolbar, buttons, and red border remain solid
            canvas_bg = self.transparent_color
            try:
                self.root.attributes("-transparentcolor", self.transparent_color)
            except tk.TclError:
                # Fallback for systems that only support alpha transparency
                try:
                    self.root.attributes("-alpha", self.window_alpha if self.window_alpha < 1.0 else 0.3)
                except tk.TclError:
                    pass
                canvas_bg = "white"

            # Top control toolbar (Play, Pause, Cancel, Region indicator)
            toolbar = tk.Frame(self.root, bg="#1e1e24", height=38)
            toolbar.pack(side="top", fill="x")

            # Play Button (Start feed)
            self.play_btn = tk.Button(
                toolbar,
                text="▶ Play",
                command=self.play,
                bg="#2ecc71",
                fg="white",
                activebackground="#27ae60",
                activeforeground="white",
                font=("Arial", 9, "bold"),
                relief="flat",
                padx=10,
                pady=2,
                cursor="hand2"
            )
            self.play_btn.pack(side="left", padx=4, pady=4)

            # Pause Button (Stop feed)
            self.pause_btn = tk.Button(
                toolbar,
                text="⏸ Pause",
                command=self.pause,
                bg="#f39c12",
                fg="white",
                activebackground="#d68910",
                activeforeground="white",
                font=("Arial", 9, "bold"),
                relief="flat",
                padx=10,
                pady=2,
                cursor="hand2"
            )
            self.pause_btn.pack(side="left", padx=4, pady=4)

            # Cancel / Close Button (Cancel window without crashing)
            cancel_btn = tk.Button(
                toolbar,
                text="✕ Cancel",
                command=self.close,
                bg="#e74c3c",
                fg="white",
                activebackground="#c0392b",
                activeforeground="white",
                font=("Arial", 9, "bold"),
                relief="flat",
                padx=10,
                pady=2,
                cursor="hand2"
            )
            cancel_btn.pack(side="left", padx=4, pady=4)

            # Status Label
            self.status_label = tk.Label(
                toolbar,
                text="● Ready",
                bg="#1e1e24",
                fg="#2ecc71",
                font=("Arial", 9, "bold"),
                padx=8
            )
            self.status_label.pack(side="left", padx=4, pady=4)

            # Region Coordinates Label
            self.region_label = tk.Label(
                toolbar,
                text="Region: [---]",
                bg="#1e1e24",
                fg="#ecf0f1",
                font=("Consolas", 9)
            )
            self.region_label.pack(side="right", padx=10, pady=4)

            # Canvas viewport representing the capture area
            # The canvas area inside the red border is 100% transparent to clearly view ground
            self.canvas = tk.Canvas(
                self.root,
                highlightthickness=3,
                highlightbackground="#e74c3c",
                bg=canvas_bg
            )
            self.canvas.pack(fill="both", expand=True)

            # Allow dragging the window by clicking and dragging anywhere on the toolbar
            def start_move(event):
                self._drag_x = event.x
                self._drag_y = event.y

            def on_move(event):
                if hasattr(self, "_drag_x") and hasattr(self, "_drag_y"):
                    deltax = event.x - self._drag_x
                    deltay = event.y - self._drag_y
                    x = self.root.winfo_x() + deltax
                    y = self.root.winfo_y() + deltay
                    self.root.geometry(f"+{x}+{y}")

            toolbar.bind("<ButtonPress-1>", start_move)
            toolbar.bind("<B1-Motion>", on_move)

            # Intercept window close protocols cleanly:
            # Clicking [X] button or pressing Escape triggers self.close() safely
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            self.root.bind("<Escape>", lambda e: self.close())

            self._is_alive = True
            self.running = False
            self.is_paused = False
            self._capture_error_logged = False

            self._safe_update()
            self._update_region_display()

            logger.info("Capture window created successfully. Move and resize window over the target game area.")
        except Exception as e:
            logger.error(f"Error creating capture window: {e}")
            self._cleanup_handles()

    def is_active(self) -> bool:
        """Check if the capture window exists and is alive."""
        return (
            self.root is not None
            and self._is_alive
            and self._check_window_exists()
        )

    def _check_window_exists(self) -> bool:
        if self.root is None:
            return False
        try:
            return bool(self.root.winfo_exists())
        except (tk.TclError, Exception):
            return False

    def _safe_update(self) -> bool:
        """
        Safely update the Tkinter event loop without throwing uncaught TclErrors
        if the window was closed.
        """
        if getattr(self, "_close_requested", False) or not self.is_active():
            if getattr(self, "_close_requested", False):
                self._cleanup_handles()
            return False

        try:
            self.root.update_idletasks()
            self.root.update()
            return True
        except (tk.TclError, RuntimeError) as e:
            logger.debug(f"Window update intercepted closure: {e}")
            self._handle_window_closed()
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during GUI update: {e}")
            return False

    def _update_region_display(self) -> None:
        """Update region text in toolbar with current canvas coordinates."""
        if not self.is_active():
            return
        try:
            region = self.get_region()
            if self.region_label and self.region_label.winfo_exists():
                self.region_label.config(
                    text=f"X:{region['left']} Y:{region['top']} | {region['width']}x{region['height']}"
                )
        except Exception:
            pass

    def play(self) -> None:
        """Start or resume continuous frame streaming."""
        if not self.is_active():
            logger.warning("Cannot start stream: capture window is not open. Calling select_region()...")
            self.select_region()

        self.running = True
        self.is_paused = False

        if self.status_label and self.is_active():
            try:
                self.status_label.config(text="● Streaming", fg="#2ecc71")
            except Exception:
                pass

        logger.info("Streaming feed started [PLAY]. Continuous frames active.")

    def pause(self) -> None:
        """Pause continuous frame streaming without closing window."""
        self.is_paused = True

        if self.status_label and self.is_active():
            try:
                self.status_label.config(text="⏸ Paused", fg="#f39c12")
            except Exception:
                pass

        logger.info("Streaming feed paused [PAUSE]. Frame generation halted temporarily.")

    def stop(self) -> None:
        """Stop streaming loop without destroying window."""
        self.running = False
        self.is_paused = False

        if self.status_label and self.is_active():
            try:
                self.status_label.config(text="■ Stopped", fg="#95a5a6")
            except Exception:
                pass

        logger.info("Streaming feed stopped [STOP].")

    def close(self) -> None:
        """
        Cleanly shut down capture and destroy Tkinter window without crashing the host process.
        """
        if not self._is_alive and self.root is None:
            return

        logger.info("Window close/cancel requested. Performing clean shutdown...")
        self.running = False
        self.is_paused = False
        self._is_alive = False
        self._close_requested = True

        if self.video_recorder and self.video_recorder.is_recording:
            self.stop_recording()

        self._cleanup_handles()
        logger.info("Capture window and resources closed cleanly.")

    def _handle_window_closed(self) -> None:
        """Handle unexpected or OS-level window termination."""
        self.running = False
        self.is_paused = False
        self._is_alive = False
        self._close_requested = True
        self._cleanup_handles()

    def _cleanup_handles(self) -> None:
        """Destroy tkinter root safely and reset all widget references."""
        root = self.root
        self.root = None
        self.canvas = None
        self.status_label = None
        self.region_label = None
        self.play_btn = None
        self.pause_btn = None

        if root is not None:
            try:
                root.destroy()
            except (tk.TclError, Exception):
                pass

    def get_region(self) -> Dict[str, int]:
        """
        Get the current capture coordinates and dimensions.
        Targets the canvas viewport area below the toolbar so controls are excluded.
        """
        if not self.is_active():
            return self.last_region

        self._safe_update()

        try:
            # Capture the canvas viewport (excludes the toolbar buttons)
            target = self.canvas if self.canvas is not None else self.root
            left = max(0, target.winfo_rootx())
            top = max(0, target.winfo_rooty())
            width = max(1, target.winfo_width())
            height = max(1, target.winfo_height())

            self.last_region = {
                "left": left,
                "top": top,
                "width": width,
                "height": height
            }
            return self.last_region
        except (tk.TclError, Exception) as e:
            logger.debug(f"Could not compute current region: {e}")
            return self.last_region

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single screen frame from the current window location.
        Returns:
            np.ndarray (RGB image) or None if capture fails or window closed.
        """
        if not self.is_active():
            return None

        self._safe_update()

        try:
            region = self.get_region()
            if region["width"] <= 0 or region["height"] <= 0:
                logger.warning(f"Invalid capture region size: {region}")
                return None

            screenshot = pyautogui.screenshot(
                region=(
                    region["left"],
                    region["top"],
                    region["width"],
                    region["height"]
                )
            )

            frame = np.array(screenshot)

            # If capture previously failed, log recovery
            if self._capture_error_logged:
                logger.info("Screen capture restored successfully.")
                self._capture_error_logged = False

            # If recording active, write frame
            if self.video_recorder and self.video_recorder.is_recording:
                self.video_recorder.write_frame(frame)

            return frame
        except Exception as e:
            if not self._capture_error_logged:
                logger.error(f"Error during screen capture: {e}")
                self._capture_error_logged = True
            return None

    def start(self, fps_limit: Optional[float] = None) -> Generator[np.ndarray, None, None]:
        """
        Generator continuously yielding captured frames.
        - Respects Play / Pause states.
        - Gracefully halts without raising exceptions when window is cancelled or closed.
        - Limits CPU consumption while paused.
        """
        if not self.is_active():
            logger.info("Initializing capture window for stream...")
            self.select_region()

        self.running = True
        self.is_paused = False

        frame_delay = (1.0 / fps_limit) if (fps_limit and fps_limit > 0) else 0.001
        logger.info("Frame streaming generator started.")

        try:
            while self.running and self.is_active():
                # Process UI events
                if not self._safe_update():
                    break

                self._update_region_display()

                # Handle Pause state
                if self.is_paused:
                    time.sleep(0.05)
                    continue

                frame = self.capture()
                if frame is not None:
                    yield frame
                else:
                    if not self.is_active():
                        break
                    time.sleep(0.02)

                if frame_delay > 0:
                    time.sleep(frame_delay)

        except KeyboardInterrupt:
            logger.info("Stream interrupted by user keyboard.")
        except Exception as e:
            logger.error(f"Stream error encountered: {e}")
        finally:
            self.stop()
            if getattr(self, "_close_requested", False) or not self._is_alive:
                self._cleanup_handles()
            logger.info("Frame streaming generator closed cleanly.")

    def start_recording(self, output_path: Optional[str] = None, fps: float = 30.0) -> bool:
        """
        Start recording captured frames to a video file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("logs", "recordings", f"dino_stream_{timestamp}.mp4")

        region = self.get_region()
        w, h = region["width"], region["height"]
        if w <= 0 or h <= 0:
            w, h = 800, 300

        self.video_recorder = VideoRecorder(output_path=output_path, fps=fps)
        return self.video_recorder.start(w, h)

    def stop_recording(self) -> Optional[str]:
        """
        Stop video recording and finalize the file.
        """
        if self.video_recorder and self.video_recorder.is_recording:
            saved_path = self.video_recorder.stop()
            return saved_path
        return None

    def save_frame(self, frame: np.ndarray, filepath: str) -> bool:
        """
        Save a single frame (RGB format) to an image file.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            if cv2 is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr)
            else:
                from PIL import Image
                Image.fromarray(frame).save(filepath)
            logger.info(f"Frame saved successfully to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False


  