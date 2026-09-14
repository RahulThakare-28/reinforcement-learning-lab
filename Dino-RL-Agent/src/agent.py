
from config import MODEL_PATH
from screen import capture_game
from observation import get_observation
from controller import execute_action

import time
from stable_baselines3 import DQN

# Load Your Trained Model
model = DQN.load(MODEL_PATH)

frame = capture_game()

observation = get_observation(frame)

action, _ = model.predict(
    observation,
    deterministic=True
)


# agent 
def run_agent():

    global previous_dino_y

    previous_dino_y = None

    while True:

        frame = capture_game()

        observation = get_observation(frame)

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        execute_action(action)

        print(
            f"Obs={observation} | "
            f"Action={int(action)}"
        )

        time.sleep(0.03)