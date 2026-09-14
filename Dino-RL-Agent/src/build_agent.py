# train DQN on new environment version 2
from stable_baselines3 import DQN
from dino_env import DinoEnv

env = DinoEnv()

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10_000,
    learning_starts=1_000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1
)

if __name__ == "__main__":

    # train the model
    model.learn(total_timesteps=100_0000)

    # save the model
    model.save("../models/dino_dqn_v2")