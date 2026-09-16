import logging
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

    def __init__(
        self,
        failsafe: bool = True,
        pause_on_start: bool = True,
        log_level: int = logging.INFO
    ):
        super().__init__(
            name="EnvV2Controller",
            failsafe=failsafe,
            pause_on_start=pause_on_start,
            log_level=log_level
        )

    def execute_action(self, action: Union[int, float]) -> bool:
        """
        Execute an action for Environment V2.

        Parameters:
            action (int or float):
                0: DUCK (presses down)
                1: JUMP (presses up)
                2: DO NOTHING (no key press)

        Returns:
            bool: True if action was executed successfully, False otherwise.
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
