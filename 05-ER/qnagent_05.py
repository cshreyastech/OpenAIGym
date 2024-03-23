import random
import numpy as np

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from collections import deque
from agent_class_05 import Agent

class QNAgent(Agent):
  def __init__(self, env, discount_rate=0.97, learning_rate=0.001):
    super().__init__(env)
    self.state_size = env.observation_space.n
    print("State size:", self.state_size)
    
    self.eps = 1.0
    self.discount_rate = discount_rate
    self.learning_rate = learning_rate
    self.build_model()

    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.replay_buffer = deque(maxlen=1000)
    
  def build_model(self):
    tf.reset_default_graph()
    self.state_in = tf.placeholder(tf.int32, shape=[None]) # (1, None)
    self.action_in = tf.placeholder(tf.int32, shape=[None]) # (1,  None)
    self.target_in = tf.placeholder(tf.float32, shape=[None]) # (1,  None)

    self.state = tf.one_hot(self.state_in, depth=self.state_size) # (?, state_size)
    self.action = tf.one_hot(self.action_in, depth=self.action_size) # (?, action_size)
    self.q_state = tf.layers.dense(inputs=self.state, units=self.action_size, name="q_table") # (1, action_size)
    # m = tf.multiply(self.q_state, self.action) # (1, action_size) element-wise
    
    self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis=1) # (1, )
    self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
      
  def get_action(self, state):
    q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]}) # (1, action_size)

    action_greedy = np.argmax(q_state)
    action_random = super().get_action(state)
    return action_random if random.random() < self.eps else action_greedy
  
  def train(self, experience, batch_size=50):
    self.replay_buffer.append(experience)
    samples = random.choices(self.replay_buffer, k=batch_size)
    # state, action, next_state, reward, done = ([exp] for exp in experience)
    state, action, next_state, reward, done = (list(col) for col in zip(experience, *samples))
    # print("state", len(state))

    q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state}) # (1, action_size)

    q_next[done] = np.zeros([self.action_size])
    q_target = reward + self.discount_rate * np.max(q_next, axis=1)
    
    feed = {self.state_in: state, self.action_in: action, self.target_in: q_target}
    self.sess.run(self.optimizer, feed_dict=feed)
    
    if experience[4]:
      self.eps = self.eps * 0.99
          
  # def __del__(self):
  #   self.sess.close()
