"""
Execution runner for DinoV1 RL Agent.
Runs the trained DQN agent with live screen observation and EnvV2 keyboard control.
"""

import os
import sys

# ── Path resolution: ensures project root and src are importable ──────────────
try:
    _here = os.path.abspath(os.path.dirname(__file__))
except NameError:
    from pathlib import Path
    _cwd = Path.cwd().resolve()
    _here = str(_cwd)

PROJECT_ROOT = os.path.abspath(os.path.join(_here, ".."))
SRC_PATH = os.path.abspath(os.path.join(_here, "..", "src"))

for p in [PROJECT_ROOT, SRC_PATH, _here]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Import trained model path from agents_config ──────────────────────────────
try:
    from .agents_config import DINOV1
except (ImportError, ValueError):
    try:
        from agents_config import DINOV1
    except ImportError:
        from src.agents.agents_config import DINOV1

# ── Import run_dinoV1 controller function ─────────────────────────────────────
try:
    from src.agents.dinoV1.dinoV1_control import run_dinoV1
except ImportError:
    
    # pyrefly: ignore [missing-import]
    from ..agents.dinoV1.dinoV1_control import run_dinoV1


if __name__ == "__main__":
    print(f"[RUNNER] Starting DinoV1 agent with model config: {DINOV1}")
    run_dinoV1(model_path=DINOV1)
