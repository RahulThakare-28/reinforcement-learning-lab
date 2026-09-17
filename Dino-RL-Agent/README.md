# 🦖 Dino RL Agent

A hands-on **Reinforcement Learning (RL) project** that teaches an agent to play the Google Chrome Offline Dino Game by observing the game state, selecting actions, and learning from rewards.

The project is being developed incrementally: start with a simplified custom environment to understand RL, then progressively connect the learned agent to the real Chrome Dino Game.

---

## 🎯 Project Objective

The primary objective is to apply Reinforcement Learning concepts to a real-world interactive problem.

Instead of manually programming rules such as:

```text
IF obstacle is near:
    jump
```

the goal is to develop an agent that learns:

```text
Observe the environment
        ↓
Understand the current state
        ↓
Select an action
        ↓
Receive reward
        ↓
Learn from the experience
        ↓
Make better decisions
```

The final system should be capable of observing obstacles in the Dino Game and deciding when to perform an action to avoid them.

---

## 🧠 Learning Objective

This project is also a practical continuation of the Reinforcement Learning fundamentals being studied.

The project will connect theoretical RL concepts with an actual working system:

* Agent
* Environment
* State
* Observation
* Action
* Reward
* Episode
* Policy
* Value / Q-value
* Exploration vs. exploitation
* Q-Learning
* Deep Q-Network (DQN)
* Experience Replay
* Target Network
* Deep Reinforcement Learning
* CNN + RL for visual observations

The learning approach is intentionally **problem-first**:

> Build the system → encounter a technical problem → learn the specific concept required to solve it → continue building.

This follows the project's practical learning philosophy of starting with a real-world problem and studying deeper theory when a technical wall is encountered.

---

# 🎮 Target Environment

The target environment is the **Google Chrome Offline Dino Game**.

The Dino must react to obstacles appearing in its path.

Conceptually:

```text
                 Chrome Dino Game
                        │
                        │ Observation
                        ↓
                  ┌───────────┐
                  │   Agent   │
                  └─────┬─────┘
                        │
                      Action
                        ↓
                ┌───────────────┐
                │   Dino Game   │
                └───────┬───────┘
                        │
                     Reward
                        │
                        └──────────→ Agent
```

---

# 🎯 Action Space

The initial RL formulation uses a discrete action space.

```text
Action 0 → DOWN / NO JUMP
Action 1 → UP / JUMP
```

Therefore:

```text
Action Space = Discrete(2)
```

The action representation may be refined later if the real game interaction requires additional control behavior.

---

# 👁️ Observation

The agent needs information about the current game situation.

The project will progressively move from a simplified structured observation toward visual observation.

### Initial observation

The custom environment can provide features such as:

```text
distance_to_obstacle
obstacle_height
dino_height
dino_velocity
```

Example:

```text
State =
[
    distance_to_obstacle,
    obstacle_height,
    dino_height,
    dino_velocity
]
```

### Later observation

The real Chrome game will provide visual information:

```text
Chrome Screen
      ↓
Game Region
      ↓
Image / Frame
      ↓
Visual Processing
      ↓
Agent
      ↓
Action
```

The eventual objective is to allow the agent to make decisions from the visual game state rather than relying entirely on manually supplied game variables.

---

# 🧩 Environment

The first version will **not directly depend on Chrome**.

Instead, a custom Gymnasium environment will simulate the important parts of the Dino Game.

```text
Dino Environment

┌───────────────────────────────┐
│                               │
│       🦖              🌵       │
│                               │
└───────────────────────────────┘
```

The custom environment will define:

* Observation space
* Action space
* State transition
* Reward
* Episode termination
* Environment reset

This allows the RL algorithm to be developed and tested before introducing screen capture and keyboard automation.

---

# 🧠 RL Algorithm

The first deep RL algorithm planned for this project is:

## DQN — Deep Q-Network

DQN is a natural first choice because the initial action space is discrete.

The conceptual architecture is:

```text
Observation
     │
     ↓
Neural Network
     │
     ↓
Q-values
     │
 ┌───┴────┐
 ↓        ↓
UP       DOWN
Q₁        Q₂
 │        │
 └───┬────┘
     ↓
Highest Q-value
     ↓
Selected Action
```

The implementation will initially use **Stable-Baselines3** rather than implementing the entire DQN algorithm from scratch.

However, the internal RL concepts will be studied rather than treated as a black box.

---

# 🛠️ Technology Stack

## Development

* Python 3.11.9
* VS Code
* Python `.venv`
* Jupyter Notebook

## RL

* Gymnasium — RL environment interface
* Stable-Baselines3 — RL algorithm implementations

## Numerical / ML

* NumPy
* PyTorch

Additional libraries will be introduced only when required by a specific project stage.

The intention is to keep the initial environment simple and avoid unnecessary tooling.

---

# 🏗️ Development Strategy

The project will be developed in multiple stages.

## Phase 1 — RL Environment

Build a simple custom Dino environment.

```text
Python
  ↓
Custom Dino Environment
  ↓
Random Actions
  ↓
Environment Response
  ↓
Reward
```

### Goals

* Understand Gymnasium's environment interface
* Define observation space
* Define action space
* Implement `reset()`
* Implement `step()`
* Implement rewards
* Implement episode termination

---

## Phase 2 — Random Agent

Before training anything, test the environment with random actions.

```text
Environment
     ↓
Random Agent
     ↓
Action
     ↓
Environment
     ↓
Reward
```

This verifies that the environment itself behaves correctly.

---

## Phase 3 — DQN Agent

Introduce DQN.

```text
Observation
      ↓
     DQN
      ↓
  Q-values
      ↓
   Action
      ↓
Environment
      ↓
Reward
      ↓
Learning
```

The agent should progressively improve its behavior through repeated episodes.

---

## Phase 4 — Improve the Environment

The simplified environment will gradually become more representative of the real Dino Game.

Potential features include:

* Obstacle distance
* Obstacle width
* Obstacle height
* Dino position
* Dino velocity
* Ground obstacles
* Air obstacles
* Increasing game speed
* Multiple obstacle configurations

---

## Phase 5 — Chrome Integration

Connect the RL system to the real Chrome Dino Game.

Target architecture:

```text
┌───────────────────────┐
│   Chrome Dino Game    │
└───────────┬───────────┘
            │
            ↓
      Screen Capture
            │
            ↓
       Game Region
            │
            ↓
    Observation Extraction
            │
            ↓
          DQN
            │
            ↓
          Action
            │
            ↓
      Keyboard Input
            │
            ↓
   Chrome Dino Game
```

---

## Phase 6 — Visual RL

The final direction is to allow the agent to work from visual observations.

```text
Game Screen
     ↓
Game Region
     ↓
Image
     ↓
CNN
     ↓
Learned Representation
     ↓
DQN
     ↓
UP / DOWN
```

This phase combines previously learned **CNN concepts** with Reinforcement Learning.

---

# 🔬 Experimental Approach

The project will use controlled experiments rather than immediately building the complete system.

For example:

```text
Experiment 1
Structured state → DQN

Experiment 2
Structured state + velocity → DQN

Experiment 3
More realistic obstacle dynamics → DQN

Experiment 4
Real game state extraction → DQN

Experiment 5
Visual observation → CNN + DQN
```

For each experiment, we can compare:

* Episode reward
* Survival duration
* Number of obstacles avoided
* Training stability
* Exploration behavior
* Model performance

---

# 📊 Evaluation

The agent will be evaluated using measurable gameplay performance.

Possible metrics:

### Survival Time

How long the Dino survives before collision.

### Episode Reward

Total reward accumulated during an episode.

### Obstacles Avoided

Number of obstacles successfully passed.

### Training Performance

How performance changes across training episodes.

Example:

```text
Episode 1      →  Low survival
Episode 100    →  Improved
Episode 500    →  Better
Episode 1000   →  Stable behavior
```

The exact reward design and evaluation metrics may evolve as the environment becomes more realistic.

---

# 📁 Planned Project Structure

The structure will evolve with the project.

Initial structure:

```text
dino-rl-agent/
│
├── notebooks/
│   ├── 01_environment.ipynb
│   ├── 02_random_agent.ipynb
│   ├── 03_dqn_agent.ipynb
│   └── ...
│
├── src/
│   ├── environment/
│   │   └── dino_env.py
│   │
│   ├── agents/
│   │   └── ...
│   │
│   ├── training/
│   │   └── ...
│   │
│   └── game/
│       └── ...
│
├── models/
│
├── experiments/
│
├── requirements.txt
│
└── README.md
```

The exact structure is not fixed yet and will be refined as implementation requirements become clear.

---

# 🚦 Current Status

### Current stage

**Phase 1 — Custom RL Environment**

### Completed

* RL fundamentals studied
* Agent/environment concept understood
* Action space identified
* Initial observation concept identified
* DQN selected as the first DRL algorithm
* Development environment selected

### In Progress

* Custom Dino environment
* Gymnasium integration
* Observation-space design
* Action-space implementation

### Upcoming

* Random-agent testing
* DQN training
* Environment refinement
* Real Chrome integration
* Screen observation
* Visual RL

---

# 📚 Learning Philosophy

This is not intended to be only a "Dino game bot."

The project is being used as a practical laboratory for understanding Reinforcement Learning.

The core loop is:

```text
REAL PROBLEM
     ↓
BUILD
     ↓
TECHNICAL WALL
     ↓
LEARN REQUIRED CONCEPT
     ↓
SOLVE
     ↓
BUILD FURTHER
     ↓
NEW TECHNICAL WALL
     ↓
REPEAT
```

This approach intentionally connects theoretical concepts to implementation problems rather than studying every RL topic before building anything.

---

# 🧠 Core Learning Principle

The objective is to understand **why the system works**, not merely make the model work.

For every major component, we should be able to answer:

```text
What is it?
Why do we need it?
What problem does it solve?
What goes into it?
What comes out?
How does it interact with the rest of the system?
```

For example:

```text
Why DQN?
Why Q-values?
Why a neural network?
Why experience replay?
Why a target network?
Why exploration?
Why this reward?
Why this observation?
```

The project will therefore prioritize understanding over blindly using libraries.

---

# 🚀 Long-Term Goal

Build a complete Deep Reinforcement Learning system that can interact with the Chrome Dino Game:

```text
              🦖 DINO RL AGENT
                     │
                     ↓
              Observe Game
                     │
                     ↓
             Detect Obstacles
                     │
                     ↓
             Build Observation
                     │
                     ↓
               Neural Network
                     │
                     ↓
                  DQN
                     │
                     ↓
              Choose Action
                ↙         ↘
             UP            DOWN
                ↘         ↙
                     ↓
               Chrome Game
                     │
                     ↓
                  Reward
                     │
                     ↓
                  Learn
                     │
                     └───────────────┐
                                     │
                                     ↓
                              Better Decisions
```

The final goal is not simply to hard-code the Dino's behavior, but to demonstrate a complete **observe → act → receive feedback → learn → improve** Reinforcement Learning cycle in a real interactive environment.
