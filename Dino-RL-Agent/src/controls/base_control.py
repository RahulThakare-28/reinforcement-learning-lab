import logging
import sys
from typing import Optional
import pyautogui
from pynput import keyboard


class BaseDinoController:
    """
    Base controller providing keyboard listeners, state management,
    safety checks, and logging for Chrome Dino RL agent execution.
    """

    def __init__(
        self,
        name: str = "BaseController",
        failsafe: bool = True,
        pause_on_start: bool = True,
        log_level: int = logging.INFO
    ):
        self.name = name
        self.paused = pause_on_start
        self.quit_game = False
        self.listener: Optional[keyboard.Listener] = None

        # Configure logger
        self.logger = logging.getLogger(f"DinoControl.{self.name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [Controller] %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        # PyAutoGUI safety configuration
        pyautogui.FAILSAFE = failsafe
        self.logger.debug(f"{self.name} initialized (FAILSAFE={failsafe}).")

    def _on_key_press(self, key):
        """Handle global hotkeys for agent control."""
        try:
            if hasattr(key, "char") and key.char is not None:
                char = key.char.lower()
                if char == "s":
                    self.paused = False
                    self.logger.info("[AGENT STARTED]")
                elif char == "p":
                    self.paused = True
                    self.logger.info("[AGENT PAUSED]")
                elif char == "r":
                    self.paused = False
                    self.logger.info("[AGENT RESUMED]")
                elif char == "q":
                    self.quit_game = True
                    self.logger.info("[AGENT QUIT]")
        except Exception as e:
            self.logger.error(f"Error handling key press event: {e}")

    def start_controller(self) -> keyboard.Listener:
        """Start the keyboard listener in a background daemon thread."""
        if self.listener is None or not self.listener.is_alive():
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.daemon = True
            self.listener.start()
            self.logger.info(f"{self.name} keyboard listener started.")
        return self.listener

    start = start_controller

    def stop_controller(self) -> None:
        """Stop the keyboard listener."""
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping listener: {e}")
            finally:
                self.listener = None
                self.logger.info(f"{self.name} keyboard listener stopped.")

    stop = stop_controller

    def execute_action(self, action: int) -> bool:
        """Subclasses must implement environment-specific action execution."""
        raise NotImplementedError("Subclasses must implement execute_action.")

    def reset(self) -> None:
        """Reset state flags."""
        self.paused = True
        self.quit_game = False
        self.logger.info(f"{self.name} reset to initial state.")
