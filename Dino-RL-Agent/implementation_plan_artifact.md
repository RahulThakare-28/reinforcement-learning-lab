# Implementation Plan: Fix Anomalies A–D and Improve DinoV1 Agent

Address the 4 anomalies discovered during live Chrome Dino execution (`terminal_run_log.txt` and `obs_session_20260916_160128.jsonl`) by implementing Fixes 1 through 4 across detection, perception, controller action handling, and environment calibration/training.

## User Review Required

> [!IMPORTANT]
> - **Fix 1 & 2 (Perception)**: Eliminates the ground line from being detected as an 809 px obstacle, preventing false obstacle detection and frozen distance.
> - **Fix 3 (Control)**: Adds an airborne jump cooldown (~350ms) to stop the controller from spamming jump keypresses 30 times per second into the OS keyboard buffer.
> - **Fix 4 (Environment & Model)**: Calibrates `DinoEnvV2` base speed from an unrealistic 35 px/frame down to 18 px/frame (matching real telemetry ~16–22 px/frame), and retrains `dinoV1.zip` so it jumps at the correct distance.

---

## Review of Anomalies A to D

| Anomaly | Observed Symptoms in Telemetry | Root Cause |
|---|---|---|
| **A: Obstacle Width ~809 px** | `obs_width` is 809.0, 741.0, 663.0 px (screen width). | In [detector.py](file:///d:/GitHub/reinforcement-learning-lab/Dino-RL-Agent/src/dino_vision/detection/detector.py#L169), obstacle search ROI includes `ground_y + 10`. Morphological closing fuses the cactus with the continuous horizontal ground line into one giant contour. |
| **B: Distance Stuck at ~48–52 px** | Distance remains exactly 52.0 px across 10+ consecutive frames (`Frames 584–593`). Constant `JUMP` spamming. | Because the ground line contour begins immediately in front of Dino at $x \approx 140$, the distance $(x_{\text{obs}} - x_{\text{dino}})$ never advances. The agent panics and jumps continuously. |
| **C: Dino Contour Breakup & "Fatal Duck"** | At `[Frame 0594]`, Dino dimensions collapse to $19 \times 42$ px. Agent suddenly executes `DUCK` facing a cactus. | Partial contour fragmentation during jumping/ducking creates out-of-distribution input vector. Model hallucinates ducking against ground obstacles. |
| **D: Stagnant Velocity ($v = 11.1$)** | Velocity stays frozen at 11.1 across multiple frames despite game acceleration. | When distance is stationary, $\Delta\text{dist} \approx 0$. When new obstacles appear, $\Delta\text{dist} > 40$ triggers spawn reset without updating EMA speed. |

---

## Proposed Changes

### Component 1: Vision Detection & Perception (`src/dino_vision/`)

#### [MODIFY] [detector.py](file:///d:/GitHub/reinforcement-learning-lab/Dino-RL-Agent/src/dino_vision/detection/detector.py)
- **Fix 1 (Ground Isolation)**:
  - In `detect_obstacle()`, crop search ROI strictly above the ground line: `bottom_y = max(top_y + 10, ground_y - 2)`.
  - Zero out any residual ground rows: `roi[max(0, ground_y - 2 - top_y):, :] = 0`.
  - Reject wide contours and horizontal ground fragments:
    `if w > 140 or (w > h * 3.5 and h < 22): continue`.
  - Classify Cactus vs Bird based on base proximity: `obstacle_bottom >= ground_y - 12` is `cactus`, else `bird`.
- **Fix 2 (Dino Contour Stability)**:
  - In `detect_dino()`, filter out tiny fragments ($w < 20$ or $h < 25$).
  - If multiple candidate contours exist within the Dino ROI, merge vertically overlapping contours into a single bounding box.
  - If no valid candidate exceeds minimum anatomical dimensions, retain `self.last_dino`.

#### [MODIFY] [observation.py](file:///d:/GitHub/reinforcement-learning-lab/Dino-RL-Agent/src/dino_vision/perception/observation.py)
- **Fix 2 (Dimension Clamping & Velocity Safeguards)**:
  - In `create_observation()`:
    - Clamp `obstacle_width` to $[15.0, 100.0]$ px.
    - Clamp `obstacle_height` to $[12.0, 75.0]$ px.
    - Clamp `dino_width` to $[40.0, 75.0]$ px.
    - Clamp `dino_height` to $[30.0, 65.0]$ px.
  - In `calculate_relative_velocity()`:
    - Safeguard velocity within realistic bounds $[10.0, 45.0]$ px/frame.
    - When distance is stationary or obstacle is absent, maintain last valid smoothed speed rather than decaying to zero.

---

### Component 2: Action Control (`src/controls/`)

#### [MODIFY] [env_v2_control.py](file:///d:/GitHub/reinforcement-learning-lab/Dino-RL-Agent/src/controls/env_v2_control.py)
- **Fix 3 (Jump Cooldown & Keypress Optimization)**:
  - Add `jump_cooldown` (default: 0.35s, matching Dino airtime ~450ms).
  - If `action == 1 (JUMP)` and time since last jump $< 0.35\text{s}$, skip physical keypress to prevent OS queue saturation.
  - Support `execute_action(action, is_jumping=False)`: if agent telemetry indicates Dino is airborne, suppress repeated JUMP keypresses.
  - Ensure `DUCK (0)` and `DO_NOTHING (2)` release held keys cleanly.

---

### Component 3: Simulation Environment & Training (`src/dino_env/` & `src/agents/dinoV1/`)

#### [MODIFY] [dino_env_v2.py](file:///d:/GitHub/reinforcement-learning-lab/Dino-RL-Agent/src/dino_env/dino_env_v2.py)
- **Fix 4 (Realistic Calibration)**:
  - Adjust `self.base_speed = 18.0` (calibrated from real session median $16\text{--}22\text{ px/frame}$ instead of 35.0).
  - Adjust speed acceleration: `speed_boost = min(15.0, self.obstacles_survived * 0.4)`.
  - Calibrate `JUMP_DANGER_ZONE = 135.0` (matching real jump clearance window at $18\text{ px/frame}$).
  - Calibrate obstacle widths: `choice([25.0, 48.0, 72.0])` representing small, double, and triple cacti.

#### [MODIFY] [dinoV1_train.py](file:///d:/GitHub/reinforcement-learning-lab/Dino-RL-Agent/src/agents/dinoV1/dinoV1_train.py)
- Execute training on the recalibrated `DinoEnvV2` for 80,000–100,000 timesteps to produce an updated `models/dinoV1.zip`.

---

## Verification Plan

### Automated / Headless Verification
1. **Perception Unit Test**:
   - Run synthetic and real-frame test images through `detector.py` to verify `obstacle_width` is strictly $\le 100\text{ px}$ and ground line is never detected as an obstacle.
2. **Controller Unit Test**:
   - Test `EnvV2Controller.execute_action(1)` with rapid successive calls; verify cooldown prevents repeated execution within 350ms.
3. **Environment & Training Verification**:
   - Run `python src/agents/dinoV1/dinoV1_train.py` with 50,000+ timesteps in `.venv`.
   - Verify mean reward increases and model saves cleanly to `models/dinoV1.zip`.
4. **End-to-End Pipeline Smoke Test**:
   - Pass frames through `dinoV1_control.py` routines to verify zero regressions.
