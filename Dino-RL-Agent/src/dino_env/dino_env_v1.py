import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

# version 2 : Dynamic Dino World
class DinoEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # -------------------------
        # Actions
        # -------------------------
        # 0 = DOWN / NO JUMP
        # 1 = UP / JUMP

        self.action_space = gym.spaces.Discrete(2)

        # -------------------------
        # Observation
        # -------------------------
        #
        # [distance,
        #  obstacle_type,
        #  obstacle_height,
        #  dino_height,
        #  dino_velocity]
        #

        self.observation_space = gym.spaces.Box(
            low=np.array([
                0,
                0,
                0,
                0,
                -50
            ], dtype=np.float32),

            high=np.array([
                500,
                1,
                100,
                100,
                50
            ], dtype=np.float32),

            dtype=np.float32
        )

        # -------------------------
        # Dino state
        # -------------------------

        self.dino_height = 0
        self.dino_velocity = 0

        # -------------------------
        # Obstacle state
        # -------------------------

        self.distance_to_obstacle = 0
        self.obstacle_type = 0
        self.obstacle_height = 0

        # -------------------------
        # Physics
        # -------------------------

        self.jump_velocity = 15
        self.gravity = 2

        # -------------------------
        # Game settings
        # -------------------------

        self.obstacle_speed = 20
        self.max_distance = 500

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Dino starts on ground
        self.dino_height = 0
        self.dino_velocity = 0

        # Create first obstacle
        self._spawn_obstacle()

        return self._get_observation(), {}

    def _spawn_obstacle(self):

        # Random distance
        self.distance_to_obstacle = self.np_random.integers(
            200,
            501
        )

        # Random obstacle type
        self.obstacle_type = self.np_random.integers(
            0,
            2
        )

        # Ground obstacle
        if self.obstacle_type == 0:
            self.obstacle_height = 20

        # Air obstacle
        else:
            self.obstacle_height = 50

    def _get_observation(self):

        return np.array([
            self.distance_to_obstacle,
            self.obstacle_type,
            self.obstacle_height,
            self.dino_height,
            self.dino_velocity
        ], dtype=np.float32)

    def step(self, action):

        # -------------------------
        # 1. Apply jump
        # -------------------------

        if action == 1 and self.dino_height == 0:
            self.dino_velocity = self.jump_velocity

        # -------------------------
        # 2. Update Dino physics
        # -------------------------

        self.dino_height += self.dino_velocity

        self.dino_velocity -= self.gravity

        # -------------------------
        # 3. Ground collision
        # -------------------------

        if self.dino_height <= 0:

            self.dino_height = 0
            self.dino_velocity = 0

        # -------------------------
        # 4. Move obstacle
        # -------------------------

        self.distance_to_obstacle = max(
            0,
            self.distance_to_obstacle - self.obstacle_speed
        )

        # -------------------------
        # 5. Collision
        # -------------------------

        collision = (
            self.distance_to_obstacle <= 0
            and self.dino_height < self.obstacle_height
        )

        # -------------------------
        # 6. Reward
        # -------------------------

        if collision:

            reward = -100
            terminated = True

        else:

            #reward = 1
            if action == 1 and self.dino_height > 0:
                reward = 0.8
            else :
                reward = 1
            terminated = False

        # -------------------------
        # 7. New obstacle
        # -------------------------

        if self.distance_to_obstacle == 0 and not collision:

            self._spawn_obstacle()

        # -------------------------
        # 8. Truncation
        # -------------------------

        truncated = False

        # -------------------------
        # 9. Observation
        # -------------------------

        observation = self._get_observation()

        # -------------------------
        # 10. Info
        # -------------------------

        info = {
            "collision": collision,
            "obstacle_type": self.obstacle_type
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    
    def render(self):
        
        plt.figure(figsize=(10, 3))
    
        # Ground
        plt.plot([0, 500], [0, 0])
    
        # Dino
        dino_x = 100
    
        plt.scatter(
            dino_x,
            self.dino_height,
            s=500
        )
    
        # Obstacle
        obstacle_x = (
            dino_x + self.distance_to_obstacle
        )
    
        plt.scatter(
            obstacle_x,
            self.obstacle_height / 2,
            s=500
        )
    
        # Labels
        plt.xlim(0, 500)
        plt.ylim(0, 100)
    
        plt.xlabel("Horizontal Position")
        plt.ylabel("Height")
    
        plt.title("Dino RL Environment")
    
        plt.show()