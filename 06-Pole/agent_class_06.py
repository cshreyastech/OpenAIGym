import random
import numpy as np
import gymnasium as gym
import numpy as np

class HillClimbingAgent():
  def __init__(self, env):
    self.state_dim = env.observation_space.shape
    self.action_size = env.action_space.n
    self.build_model()
      
  def build_model(self):
    self.weights = 1e-4*np.random.rand(*self.state_dim, self.action_size)
    self.best_reward = -np.Inf
    self.best_weights = np.copy(self.weights)
    self.noise_scale = 1e-2
      
  def get_action(self, state):
    p = np.dot(state, self.weights)
    action = np.argmax(p)
    return action
  
  def update_model(self, reward):
    if reward >= self.best_reward:
      self.best_reward = reward
      self.best_weights = np.copy(self.weights)
      self.noise_scale = max(self.noise_scale/2, 1e-3)
    else:
      self.noise_scale = min(self.noise_scale*2, 2)
        
    self.weights = self.best_weights + self.noise_scale * np.random.rand(*self.state_dim, self.action_size)