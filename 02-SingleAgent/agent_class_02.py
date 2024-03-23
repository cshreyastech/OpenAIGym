import random
import numpy as np
import gymnasium as gym

class Agent():
  def __init__(self, env):
    self.is_discrete = \
      type(env.action_space) == gym.spaces.discrete.Discrete

    if self.is_discrete:
      self.action_size = env.action_space.n
      print("Discrate Action size:", self.action_size)
    else:
      self.action_low = env.action_space.low
      self.action_high = env.action_space.high
      self.action_shape = env.action_space.shape
      print("Continious Action range:", self.action_low, self.action_high)

  def get_action(self, state):
    if self.is_discrete:
      action = random.choice(range(self.action_size))
    else:
      action = np.random.uniform(low=self.action_low, 
                                high=self.action_high, 
                                size=self.action_shape)
    return action