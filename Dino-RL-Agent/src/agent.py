#Dino-RL-Agent/src/agent2.py
# -- verion 2 ---
import time
import numpy as np

from stable_baselines3 import DQN

from screen import capture_game
from observation import get_observation
from config import MODEL_PATH

import controller





# ============================================================
# Load trained model
# ============================================================

model = DQN.load(
    MODEL_PATH
)


# ============================================================
# Agent
# ============================================================

def run_agent():

    # Start keyboard controller
    controller.start_controller()

    print("\n================================")
    print("       DINO RL AGENT")
    print("================================")
    print("S = Start")
    print("P = Pause")
    print("R = Resume")
    print("Q = Quit")
    print("================================\n")

    while not controller.quit_game:

        # ----------------------------------------
        # PAUSED
        #
        # Agent stays alive in background.
        # It does NOT terminate.
        # ----------------------------------------

        if controller.paused:

            time.sleep(0.05)

            continue


        # ----------------------------------------
        # PLAYING
        # ----------------------------------------

        frame = capture_game()

        observation = get_observation(
            frame
        )


        # ----------------------------------------
        # DQN prediction
        # ----------------------------------------

        action, _ = model.predict(
            observation,
            deterministic=True
        )


        # ----------------------------------------
        # Execute DQN action
        #
        # 1 = UP
        # 0 = DOWN / DUCK
        # ----------------------------------------

        controller.execute_action(
            action
        )


        print(
            f"Obs={np.round(observation, 2)} | "
            f"Action={int(action)}"
        )


        time.sleep(0.03)


    # ----------------------------------------
    # Q pressed
    # ----------------------------------------

    print("\nAgent terminated.")


