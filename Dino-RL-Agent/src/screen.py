
# import pyautogui
# import cv2
# import numpy as np

# from config import GAME_REGION

# # Screen Capture
# def capture_game():
#     return pyautogui.screenshot(
#         region=(
#             GAME_REGION["left"],
#             GAME_REGION["top"],
#             GAME_REGION["width"],
#             GAME_REGION["height"]
#         )
#     )

# #frame = capture_game()
# # print(frame.size)
# # frame

# # image preprocessing
# def preprocess_frame(frame):

#     img = np.array(frame)

#     gray = cv2.cvtColor(
#         img,
#         cv2.COLOR_RGB2GRAY
#     )

#     _, binary = cv2.threshold(
#         gray,
#         100,
#         255,
#         cv2.THRESH_BINARY_INV
#     )

#     return img, gray, binary

#----- verion 2 -----
import pyautogui
import numpy as np
import cv2


GAME_REGION = {
    "left": 950,
    "top": 200,
    "width": 950,
    "height": 300
}


def capture_game():
    return pyautogui.screenshot(
        region=(
            GAME_REGION["left"],
            GAME_REGION["top"],
            GAME_REGION["width"],
            GAME_REGION["height"]
        )
    )


def preprocess_frame(frame):
    """
    Convert the Chrome Dino screenshot into a foreground mask.

    Output:
        img     -> original RGB image
        gray    -> grayscale image
        binary  -> foreground objects are WHITE (255)
                   background is BLACK (0)

    The polarity is detected automatically.
    """

    img = np.array(frame)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )
    else:
        gray = img.copy()

    # --------------------------------------------------------
    # Chrome Dino normally uses only black/white pixels.
    #
    # Determine whether the background is black or white.
    # Background occupies most of the screen.
    # --------------------------------------------------------

    mean_value = gray.mean()

    if mean_value < 128:
        # Black background -> white objects
        binary = cv2.threshold(
            gray,
            128,
            255,
            cv2.THRESH_BINARY
        )[1]

    else:
        # White background -> black objects
        binary = cv2.threshold(
            gray,
            128,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

    return img, gray, binary