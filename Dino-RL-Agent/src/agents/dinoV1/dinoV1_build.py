"""
DinoV1 Model Factory & Architecture Definition.

Responsible strictly for model construction and hyperparameter configuration
for the Dino DQN v1 reinforcement learning agent.
"""

from typing import Optional, Any, Dict
import gymnasium as gym
from stable_baselines3 import DQN

# Default hyperparameters for DinoV1 DQN architecture
DEFAULT_DINOV1_PARAMS: Dict[str, Any] = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "buffer_size": 10_000,
    "learning_starts": 1_000,
    "batch_size": 32,
    "gamma": 0.99,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.05,
    "verbose": 1,
}


def build_model(
    env: Optional[gym.Env] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> DQN:
    """
    Construct and return a configured Stable-Baselines3 DQN model.

    Parameters:
        env (gym.Env, optional): Gymnasium environment instance (e.g., DinoEnvV2).
        custom_params (dict, optional): Dictionary of hyperparameters to override defaults.
        **kwargs: Direct keyword arguments to override hyperparameters.

    Returns:
        DQN: Configured Stable-Baselines3 DQN agent instance.
    """
    params = DEFAULT_DINOV1_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    params.update(kwargs)

    return DQN(env=env, **params)


__all__ = [
    "DEFAULT_DINOV1_PARAMS",
    "build_model",
]
