


# # action controller
# import pyautogui


# def execute_action(action):
#     pyautogui.FAILSAFE = True
    
#     action = int(action)

#     if action == 1:
#         pyautogui.press("up")

#     elif action == 0:
#         pyautogui.press("down")

# -- version 2 ---
# v2
import pyautogui
from pynput import keyboard

# class Controller:
#     # Agent starts in paused state
#     def __init__(self) : 
#         pass
paused = True
quit_game = False


def on_key_press(key):
    global paused
    global quit_game

    try:
        key = key.char.lower()

        # ----------------------------------------
        # S = START agent
        # ----------------------------------------
        if key == "s":
            paused = False
            print("\n[AGENT STARTED]")

        # ----------------------------------------
        # P = PAUSE agent
        # ----------------------------------------
        elif key == "p":
            paused = True
            print("\n[AGENT PAUSED]")

        # ----------------------------------------
        # R = RESUME agent
        # ----------------------------------------
        elif key == "r":
            paused = False
            print("\n[AGENT RESUMED]")

        # ----------------------------------------
        # Q = QUIT agent
        # ----------------------------------------
        elif key == "q":
            quit_game = True
            print("\n[AGENT QUIT]")

    except AttributeError:
        pass
    
    
def start_controller():

    listener = keyboard.Listener(
        on_press=on_key_press
    )

    listener.daemon = True
    listener.start()

    return listener


def execute_action(action):

    action = int(action)

    # DQN action 1 = UP / JUMP
    if action == 1:
        pyautogui.press("up")

    # DQN action 0 = DOWN / DUCK
    elif action == 0:
        pyautogui.press("down")

    # DQN action 2 = DO NOTHING
    elif action == 2:
        pass