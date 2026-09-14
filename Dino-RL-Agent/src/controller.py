


# action controller
import pyautogui


def execute_action(action):
    pyautogui.FAILSAFE = True
    
    action = int(action)

    if action == 1:
        pyautogui.press("up")

    elif action == 0:
        pyautogui.press("down")