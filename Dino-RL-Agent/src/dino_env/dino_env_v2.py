import gymnasium as gym
import numpy as np


class DinoEnvV2(gym.Env):
    """ 
    Dino Environment Version 2 (Calibrated from Real Chrome Dino Game Vision Logs).

    Observation Vector (7 elements):
        0: distance         - Horizontal clearance to obstacle (0.0 to 600.0)
        1: velocity         - Relative approaching velocity (0.0 to 250.0)
        2: dino_height      - Dino bounding box height (33.0 ducking, 55.0 standing)
        3: dino_width       - Dino bounding box width (52.0 standing, 70.0 ducking)
        4: obstacle_height  - Obstacle bounding box height (0.0 to 150.0)
        5: obstacle_width   - Obstacle bounding box width (0.0 to 800.0)
        6: obstacle_type    - Type code: 0.0 = Bird, 1.0 = Cactus, -1.0 = None

    Action Space:
        0: Duck (DOWN)
        1: Jump (UP)
        2: Do Nothing (NONE)

    Physics:
        - When jumping, Dino receives initial upward velocity and is pulled down by gravity.
        - Peak jump height is calibrated higher than maximum cactus height (~85 px vs ~60 px).
        - Ducking lowers Dino height from 55 px to 33 px.

    Reward System:
        - Collision: -100.0 (episode terminates)
        - Survived obstacle (passed without collision): +1.0
        - Do nothing (action 2): +0.50
        - Unnecessary jump: -0.30
    """

    metadata = {"render_modes": ["human"]}

    # Dimensional constants calibrated from Obs/ dataset
    DINO_STAND_HEIGHT = 55.0
    DINO_STAND_WIDTH = 52.0
    DINO_DUCK_HEIGHT = 33.0
    DINO_DUCK_WIDTH = 70.0

    MAX_GROUND_RANGE = 600.0
    JUMP_DANGER_ZONE = 180.0  # Threshold within which a jump is considered timely for incoming cactus

    def __init__(self):
        super().__init__()

        # ====================================================
        # ACTION SPACE (3 Actions)
        # ====================================================
        # 0 = DUCK
        # 1 = JUMP
        # 2 = DO NOTHING
        self.action_space = gym.spaces.Discrete(3)

        # ====================================================
        # OBSERVATION SPACE (7 Dimensions)
        # ====================================================
        # [distance, velocity, dino_height, dino_width,
        #  obstacle_height, obstacle_width, obstacle_type]
        self.observation_space = gym.spaces.Box(
            low=np.array([
                0.0,    # distance
                0.0,    # velocity
                0.0,    # dino_height
                0.0,    # dino_width
                0.0,    # obstacle_height
                0.0,    # obstacle_width
                -1.0    # obstacle_type (-1 = None, 0 = Bird, 1 = Cactus)
            ], dtype=np.float32),
            high=np.array([
                600.0,  # distance
                250.0,  # velocity
                100.0,  # dino_height
                120.0,  # dino_width
                150.0,  # obstacle_height
                850.0,  # obstacle_width
                1.0     # obstacle_type
            ], dtype=np.float32),
            dtype=np.float32
        )

        # ====================================================
        # DINO STATE & PHYSICS
        # ====================================================
        self.dino_y_offset = 0.0       # Vertical height above ground (0.0 = on ground)
        self.dino_vertical_velocity = 0.0
        self.is_jumping = False
        self.is_ducking = False

        self.jump_velocity = 17.0      # Calibrated: gives peak height ~85px > max cactus ~65px
        self.gravity = 1.2             # Downward acceleration per step

        self.dino_height = self.DINO_STAND_HEIGHT
        self.dino_width = self.DINO_STAND_WIDTH

        # ====================================================
        # OBSTACLE STATE
        # ====================================================
        self.distance_to_obstacle = 0.0
        self.obstacle_type = 1.0       # 0.0 = Bird, 1.0 = Cactus, -1.0 = None
        self.obstacle_height = 0.0
        self.obstacle_width = 0.0
        self.bird_altitude = 0.0       # Flight altitude for birds

        # ====================================================
        # GAME & SPEED STATE
        # ====================================================
        self.base_speed = 35.0         # Calibrated from Obs median approaching speed
        self.current_speed = self.base_speed
        self.obstacles_survived = 0
        self.steps = 0

    # ========================================================
    # RESET
    # ========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dino_y_offset = 0.0
        self.dino_vertical_velocity = 0.0
        self.is_jumping = False
        self.is_ducking = False

        self.dino_height = self.DINO_STAND_HEIGHT
        self.dino_width = self.DINO_STAND_WIDTH

        self.current_speed = self.base_speed
        self.obstacles_survived = 0
        self.steps = 0

        self._spawn_obstacle()

        return self._get_observation(), {}

    # ========================================================
    # SPAWN OBSTACLE (Calibrated from Obs folder)
    # ========================================================
    def _spawn_obstacle(self):
        # Spawn distance between 350 and 600 px (ground range 600)
        self.distance_to_obstacle = float(
            self.np_random.uniform(350.0, self.MAX_GROUND_RANGE)
        )

        # Obstacle type distribution from Obs data (~65% Cactus, ~35% Bird)
        # Type coding: 0.0 = Bird, 1.0 = Cactus
        roll = self.np_random.random()
        if roll < 0.65:
            # Cactus (Ground obstacle)
            self.obstacle_type = 1.0
            # Height in [35.0, 65.0], modal ~55-59 from Obs data
            self.obstacle_height = float(self.np_random.uniform(35.0, 65.0))
            # Width in [25.0, 75.0] (small single or multi-cactus cluster)
            self.obstacle_width = float(self.np_random.choice([28.0, 45.0, 68.0]))
            self.bird_altitude = 0.0
        else:
            # Bird (Air obstacle)
            self.obstacle_type = 0.0
            # Height in [15.0, 30.0], width in [20.0, 45.0] from Obs data
            self.obstacle_height = float(self.np_random.uniform(15.0, 30.0))
            self.obstacle_width = float(self.np_random.uniform(20.0, 45.0))
            # Bird flies at mid altitude (~40 px):
            # A standing Dino (height 55) will collide, but ducking Dino (height 33) clears it!
            self.bird_altitude = 40.0

        # Slight game speed acceleration over time
        speed_boost = min(35.0, self.obstacles_survived * 0.8)
        self.current_speed = self.base_speed + speed_boost

    # ========================================================
    # OBSERVATION VECTOR
    # ========================================================
    def _get_observation(self):
        return np.array([
            self.distance_to_obstacle,
            self.current_speed,
            self.dino_height,
            self.dino_width,
            self.obstacle_height,
            self.obstacle_width,
            self.obstacle_type
        ], dtype=np.float32)

    # ========================================================
    # STEP
    # ========================================================
    def step(self, action):
        action = int(action)
        self.steps += 1

        unnecessary_jump = False
        timely_jump = False
        timely_duck = False

        # ====================================================
        # 1. PROCESS ACTION & DINO DIMENSIONS
        # ====================================================
        if action == 0:
            # DUCK
            self.is_ducking = True
            # On ground, Dino ducks down
            if self.dino_y_offset == 0.0:
                self.dino_height = self.DINO_DUCK_HEIGHT
                self.dino_width = self.DINO_DUCK_WIDTH
            else:
                # If ducking in air, enable fast fall
                self.dino_vertical_velocity -= self.gravity * 0.8

            # Check if this duck was timely for an approaching bird
            if self.obstacle_type == 0.0 and self.distance_to_obstacle < self.JUMP_DANGER_ZONE:
                timely_duck = True

        elif action == 1:
            # JUMP
            self.is_ducking = False
            self.dino_height = self.DINO_STAND_HEIGHT
            self.dino_width = self.DINO_STAND_WIDTH

            if self.dino_y_offset == 0.0:
                # Initiate jump
                self.dino_vertical_velocity = self.jump_velocity
                self.is_jumping = True

                # Evaluate jump appropriateness:
                # Necessary jump: Cactus approaching within danger zone
                if (
                    self.obstacle_type == 1.0
                    and self.distance_to_obstacle < self.JUMP_DANGER_ZONE
                ):
                    timely_jump = True
                else:
                    # Unnecessary jump: Far away or jumping when obstacle is bird/none
                    unnecessary_jump = True
            else:
                # Already in air: repeated jump press is unnecessary
                unnecessary_jump = True

        elif action == 2:
            # DO NOTHING
            self.is_ducking = False
            self.dino_height = self.DINO_STAND_HEIGHT
            self.dino_width = self.DINO_STAND_WIDTH

        # ====================================================
        # 2. DINO PHYSICS (GRAVITY & KINEMATICS)
        # ====================================================
        if self.is_jumping or self.dino_y_offset > 0.0:
            self.dino_y_offset += self.dino_vertical_velocity
            self.dino_vertical_velocity -= self.gravity

            # Ground collision check
            if self.dino_y_offset <= 0.0:
                self.dino_y_offset = 0.0
                self.dino_vertical_velocity = 0.0
                self.is_jumping = False
                if self.is_ducking:
                    self.dino_height = self.DINO_DUCK_HEIGHT
                    self.dino_width = self.DINO_DUCK_WIDTH

        # ====================================================
        # 3. MOVE OBSTACLE
        # ====================================================
        self.distance_to_obstacle = max(
            0.0,
            self.distance_to_obstacle - self.current_speed
        )

        # ====================================================
        # 4. COLLISION DETECTION
        # ====================================================
        collision = False

        if self.distance_to_obstacle <= 0.0:
            if self.obstacle_type == 1.0:
                # CACTUS: Dino must jump higher than cactus height
                if self.dino_y_offset < self.obstacle_height:
                    collision = True

            elif self.obstacle_type == 0.0:
                # BIRD: Mid-air obstacle
                # If Dino jumps into bird or stands tall -> collision!
                # If Dino ducks on ground (dino_height <= 35.0 and dino_y_offset == 0) -> safe!
                if self.dino_y_offset > 0.0 or self.dino_height > 35.0:
                    collision = True

        # ====================================================
        # 5. REWARD CALCULATION
        # ====================================================
        if collision:
            reward = -100.0
            terminated = True
        else:
            terminated = False

            # Reward components according to user specifications:
            if action == 2:
                # Do nothing: +0.50 points
                reward = 0.50
            elif action == 1:
                if unnecessary_jump:
                    # Unnecessary jump: -0.30 points
                    reward = -0.30
                else:
                    # Well-timed jump over cactus
                    reward = 0.75
            elif action == 0:
                # Duck
                if timely_duck:
                    reward = 0.75
                else:
                    reward = 0.20

            # Obstacle successfully survived: +1.0 point
            if self.distance_to_obstacle == 0.0 and not collision:
                reward += 1.0
                self.obstacles_survived += 1
                self._spawn_obstacle()

        truncated = False
        observation = self._get_observation()

        info = {
            "collision": collision,
            "obstacle_type": self.obstacle_type,
            "obstacles_survived": self.obstacles_survived,
            "dino_y_offset": self.dino_y_offset,
            "dino_height": self.dino_height,
            "current_speed": self.current_speed,
        }

        return observation, reward, terminated, truncated, info