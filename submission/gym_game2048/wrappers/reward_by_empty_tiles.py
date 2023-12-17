import gymnasium as gym
import numpy as np

class RewardByEmptyTiles(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = np.count_nonzero(obs == 0)
        return obs, reward, terminated, truncated, info