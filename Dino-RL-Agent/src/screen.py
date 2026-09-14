
import pyautogui
import cv2
import numpy as np

from config import GAME_REGION

# Screen Capture
def capture_game():
    return pyautogui.screenshot(
        region=(
            GAME_REGION["left"],
            GAME_REGION["top"],
            GAME_REGION["width"],
            GAME_REGION["height"]
        )
    )

#frame = capture_game()
# print(frame.size)
# frame

# image preprocessing
def preprocess_frame(frame):

    img = np.array(frame)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    return img, gray, binary