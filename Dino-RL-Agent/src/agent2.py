# Dino-RL-Agent/src/agent2.py
# Version 3
# Trained DQN + Dino Vision Pipeline

import time
import numpy as np

from stable_baselines3 import DQN

from .config import MODEL_PATH
#import controller
from src import controller

# Dino Vision modules
from .dino_vision.capture.screen_capture import ScreenCapture
from .dino_vision.preprocessing.preprocessor import Preprocessor
from .dino_vision.detection.detector import Detector
from .dino_vision.perception.observation import ObservationManager


# ============================================================
# Load trained model
# ============================================================

model = DQN.load(MODEL_PATH)


# ============================================================
# Create Dino Vision pipeline
# ============================================================

capture = ScreenCapture()

preprocessor = Preprocessor(
    auto_invert=True
)

detector = Detector()

obs_manager = ObservationManager()


# ============================================================
# Agent
# ============================================================

def run_agent():

    # Start keyboard controller
    controller.start_controller()

    print("\n================================")
    print("       DINO RL AGENT")
    print("================================")
    print("P = Pause / Resume")
    print("Q = Quit")
    print("================================\n")

    # --------------------------------------------------------
    # Select Chrome Dino game region
    # --------------------------------------------------------

    print("[INFO] Select the Chrome Dino game region.")

    capture.select_region()

    print("[INFO] Starting Dino Vision...")


    # ========================================================
    # Main loop
    # ========================================================

    try:

        for frame in capture.start(fps_limit=30):

            # ------------------------------------------------
            # Q = Quit
            # ------------------------------------------------

            if controller.quit_game:
                break


            # ------------------------------------------------
            # P = Pause
            # ------------------------------------------------

            if controller.paused:

                time.sleep(0.05)

                continue


            # =================================================
            # 1. PREPROCESS
            # =================================================

            prep_data = preprocessor.process(frame)

            binary = prep_data["binary"]


            # =================================================
            # 2. DETECTION
            # =================================================

            detections = detector.detect(
                binary
            )


            # =================================================
            # 3. OBSERVATION
            # =================================================

            observation_object = (
                obs_manager.create_observation(
                    detections
                )
            )


            # =================================================
            # 4. CONVERT TO DQN INPUT
            # =================================================

            observation = np.asarray(
                observation_object.to_array(),
                dtype=np.float32
            )


            # =================================================
            # 5. DQN PREDICTION
            # =================================================

            action, _ = model.predict(
                observation,
                deterministic=True
            )


            # =================================================
            # 6. EXECUTE ACTION
            #
            # 1 = UP
            # 0 = DOWN / DUCK
            # =================================================

            controller.execute_action(
                action
            )


            # =================================================
            # DEBUG
            # =================================================

            print(
                f"Obs={np.round(observation, 2)} | "
                f"Action={int(action)}"
            )


            time.sleep(0.03)


    except KeyboardInterrupt:

        print("\n[INFO] Interrupted by user.")


    except Exception as e:

        print(
            f"\n[ERROR] Agent error: {e}"
        )

        import traceback
        traceback.print_exc()


    finally:

        capture.close()

        print("\nAgent terminated.")