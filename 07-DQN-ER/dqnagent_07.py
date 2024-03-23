import random
import numpy as np

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from collections import deque
from qnetwork_07 import QNetwork
from replaybuffer_07 import ReplayBuffer

class DQNAgent():
  def __init__(self, env):
    self.state_dim = env.observation_space.shape
    self.action_size = env.action_space.n
    self.q_network = QNetwork(self.state_dim, self.action_size)
    self.replay_buffer = ReplayBuffer(maxlen=10000)
    self.gamma = 0.97
    self.eps = 1.0
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
      
  def get_action(self, state):
    q_state = self.q_network.get_q_state(self.sess, [state])
    action_greedy = np.argmax(q_state)
    action_random = np.random.randint(self.action_size)
    action = action_random if random.random() < self.eps else action_greedy
    return action
  
  def train(self, state, action, next_state, reward, done):
    self.replay_buffer.add((state, action, next_state, reward, done))
    states, actions, next_states, rewards, dones = self.replay_buffer.sample(50)
    q_next_states = self.q_network.get_q_state(self.sess, next_states)
    q_next_states[dones] = np.zeros([self.action_size])
    q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
    self.q_network.update_model(self.sess, states, actions, q_targets)
    
    if done: self.eps = max(0.1, 0.99*self.eps)
  
  # def __del__(self):
  #   self.sess.close()
