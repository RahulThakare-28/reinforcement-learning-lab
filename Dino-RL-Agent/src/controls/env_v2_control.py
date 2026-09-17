import logging
import time
from typing import Union
import pyautogui
from .base_control import BaseDinoController


class EnvV2Controller(BaseDinoController):
    """
    Action Controller for Dino Environment Version 2 (dino_env_v2.py).

    Action Space: Discrete(3)
        0 = DUCK (DOWN)
        1 = JUMP (UP)
        2 = DO NOTHING (NONE)

    Fix 3: Jump cooldown (350 ms) prevents OS keyboard buffer saturation.
    When the agent spams JUMP at 30 fps while Dino is airborne, only the
    first keypress within a 350 ms window is forwarded to the OS. Subsequent
    JUMP requests within the cooldown window are silently dropped.
    """

    ACTION_MAP = {
        0: "down",
        1: "up",
        2: None
    }

    ACTION_NAMES = {
        0: "DUCK",
        1: "JUMP",
        2: "DO_NOTHING"
    }

    # FIX 3: Minimum interval (seconds) between two JUMP keypresses.
    # Chrome Dino airtime is ~450 ms; 350 ms prevents mid-jump re-presses.
    JUMP_COOLDOWN_SEC: float = 0.35

    def __init__(
        self,
        failsafe: bool = True,
        pause_on_start: bool = True,
        log_level: int = logging.INFO,
        jump_cooldown: float = JUMP_COOLDOWN_SEC,
    ):
        super().__init__(
            name="EnvV2Controller",
            failsafe=failsafe,
            pause_on_start=pause_on_start,
            log_level=log_level
        )
        # FIX 3: Jump cooldown state
        self._jump_cooldown: float = float(jump_cooldown)
        self._last_jump_time: float = 0.0

    def execute_action(self, action: Union[int, float], is_jumping: bool = False) -> bool:
        """
        Execute an action for Environment V2.

        Parameters:
            action (int or float):
                0: DUCK  (presses down arrow)
                1: JUMP  (presses up arrow)
                2: DO NOTHING (no key press)
            is_jumping (bool):
                Optional hint from perception layer. When True, suppresses
                additional JUMP keypresses while Dino is still airborne.

        Returns:
            bool: True if action was executed (or intentionally skipped),
                  False on error.
        """
        try:
            action_int = int(action)
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid action type: {action} ({e})")
            return False

        if action_int not in self.ACTION_MAP:
            self.logger.warning(
                f"Action {action_int} out of bounds for V2 action space Discrete(3). Allowed: [0, 1, 2]."
            )
            return False

        key_to_press = self.ACTION_MAP[action_int]
        action_name = self.ACTION_NAMES[action_int]

        # Action 2 is DO NOTHING -> no key pressed
        if key_to_press is None:
            self.logger.debug(f"Executed V2 action {action_int} ({action_name}) -> no keypress.")
            return True

        # FIX 3: Jump cooldown guard
        if action_int == 1:  # JUMP
            now = time.monotonic()
            time_since_last = now - self._last_jump_time

            if is_jumping or time_since_last < self._jump_cooldown:
                self.logger.debug(
                    f"JUMP cooldown active ({time_since_last*1000:.0f} ms since last, "
                    f"cooldown={self._jump_cooldown*1000:.0f} ms). Skipping keypress."
                )
                return True  # Intentionally skipped — not an error

            self._last_jump_time = now

        try:
            pyautogui.press(key_to_press)
            self.logger.debug(f"Executed V2 action {action_int} ({action_name}) -> pressed '{key_to_press}'")
            return True
        except pyautogui.FailSafeException:
            self.logger.critical("PyAutoGUI fail-safe triggered (mouse moved to corner). Pausing agent.")
            self.paused = True
            return False
        except Exception as e:
            self.logger.error(f"Error pressing key '{key_to_press}' for action {action_int}: {e}")
            return False


# Backward-compatible alias
DinoV2Controller = EnvV2Controller
