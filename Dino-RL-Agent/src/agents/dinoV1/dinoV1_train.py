"""
DinoV1 Agent Training Script.

Responsible for the training lifecycle of the DinoV1 agent:
1. Environment instantiation (DinoEnvV2).
2. Model construction via dinoV1_build.build_model().
3. Agent training execution (.learn()).
4. Model checkpoint saving.
"""

import os
import sys
from typing import Optional

# ── Path resolution ───────────────────────────────────────────────────────────
try:
    _here = os.path.abspath(os.path.dirname(__file__))
except NameError:
    from pathlib import Path
    _cwd = Path.cwd().resolve()
    _here = str(_cwd)

SRC_PATH = os.path.abspath(os.path.join(_here, "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", "..", ".."))

for p in [PROJECT_ROOT, SRC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from dino_env.dino_env_v2 import DinoEnvV2
except ImportError:
    from src.dino_env.dino_env_v2 import DinoEnvV2

try:
    from .dinoV1_build import build_model
except (ImportError, ValueError):
    try:
        from src.agents.dinoV1.dinoV1_build import build_model
    except ImportError:
        from dinoV1_build import build_model

try:
    from ..agents_config import DINOV1
except (ImportError, ValueError):
    try:
        from .agents_config import DINOV1
    except (ImportError, ValueError):
        try:
            from src.agents.agents_config import DINOV1
        except ImportError:
            DINOV1 = os.path.join(PROJECT_ROOT, "models", "dinoV1.zip")


def train_dinoV1(
    total_timesteps: int = 100_000,
    save_path: Optional[str] = None,
    env: Optional[DinoEnvV2] = None,
    **kwargs
):
    """
    Train the DinoV1 DQN agent on DinoEnvV2 and save the trained model.

    Parameters:
        total_timesteps (int): Number of training environment timesteps.
        save_path (str, optional): Target file path for the trained model.
        env (DinoEnvV2, optional): Custom environment instance; defaults to new DinoEnvV2().
        **kwargs: Overrides for model hyperparameters.

    Returns:
        DQN: The trained model instance.
    """
    if env is None:
        env = DinoEnvV2()

    print("=" * 60)
    print("               TRAINING DINO V1 AGENT")
    print("=" * 60)
    print(f"[INFO] Environment   : {env.__class__.__name__}")
    print(f"[INFO] Total Timesteps: {total_timesteps:,}")

    # Build model using builder factory
    model = build_model(env=env, **kwargs)

    # Train
    print("[INFO] Starting training...")
    model.learn(total_timesteps=total_timesteps)
    print("[OK] Training completed.")

    # Determine save path
    if save_path is None:
        if os.path.isabs(DINOV1):
            target_path = DINOV1
        else:
            target_path = os.path.abspath(os.path.join(PROJECT_ROOT, "models", "dinoV1"))
    else:
        target_path = save_path

    # Ensure parent directory exists
    parent_dir = os.path.dirname(target_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Strip .zip if given because SB3 adds .zip automatically
    if target_path.endswith(".zip"):
        target_path = target_path[:-4]

    model.save(target_path)
    print(f"[OK] Model saved to: {target_path}.zip")

    return model


if __name__ == "__main__":
    train_dinoV1(total_timesteps=100_000)
