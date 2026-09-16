"""Learning-library-neutral observation history for olfactory POMDPs."""

from collections import deque

import gymnasium as gym
import numpy as np


class ObservationHistory(gym.Wrapper):
    """Flatten oldest-to-newest samples; reset pads with the first observation.

    Histories are per environment and contain sensor observations only. A
    recurrent policy may instead consume the original environment directly.
    """

    def __init__(self, env, length: int = 8):
        super().__init__(env)
        if not isinstance(length, int) or length < 1:
            raise ValueError("history length must be a positive integer")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ObservationHistory requires Box observations")
        self.length = length
        self.history = deque(maxlen=length)
        self.observation_space = gym.spaces.Box(
            np.tile(env.observation_space.low.ravel(), length),
            np.tile(env.observation_space.high.ravel(), length), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.extend(obs.ravel().copy() for _ in range(self.length))
        return np.concatenate(self.history), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs.ravel().copy())
        return np.concatenate(self.history), reward, terminated, truncated, info
