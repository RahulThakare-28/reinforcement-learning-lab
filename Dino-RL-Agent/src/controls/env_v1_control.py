import logging
from typing import Union
import pyautogui
from .base_control import BaseDinoController


class EnvV1Controller(BaseDinoController):
    """
    Action Controller for Dino Environment Version 1 (dino_env.py).

    Action Space: Discrete(2)
        0 = DOWN / DUCK
        1 = UP / JUMP
    """

    ACTION_MAP = {
        0: "down",
        1: "up"
    }

    ACTION_NAMES = {
        0: "DUCK",
        1: "JUMP"
    }

    def __init__(
        self,
        failsafe: bool = True,
        pause_on_start: bool = True,
        log_level: int = logging.INFO
    ):
        super().__init__(
            name="EnvV1Controller",
            failsafe=failsafe,
            pause_on_start=pause_on_start,
            log_level=log_level
        )

    def execute_action(self, action: Union[int, float]) -> bool:
        """
        Execute an action for Environment V1.

        Parameters:
            action (int or float): 0 for DUCK/DOWN, 1 for JUMP/UP.

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
                f"Action {action_int} out of bounds for V1 action space Discrete(2). Allowed: [0, 1]."
            )
            return False

        key_to_press = self.ACTION_MAP[action_int]
        action_name = self.ACTION_NAMES[action_int]

        try:
            pyautogui.press(key_to_press)
            self.logger.debug(f"Executed V1 action {action_int} ({action_name}) -> pressed '{key_to_press}'")
            return True
        except pyautogui.FailSafeException:
            self.logger.critical("PyAutoGUI fail-safe triggered (mouse moved to corner). Pausing agent.")
            self.paused = True
            return False
        except Exception as e:
            self.logger.error(f"Error pressing key '{key_to_press}' for action {action_int}: {e}")
            return False


# Backward-compatible alias
DinoV1Controller = EnvV1Controller
