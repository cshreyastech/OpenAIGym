import random
import numpy as np
import gymnasium as gym

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class QNetwork():
  def __init__(self, state_dim, action_size, tau=0.01):
    tf.reset_default_graph()
    self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
    self.action_in = tf.placeholder(tf.int32, shape=[None])
    self.q_target_in = tf.placeholder(tf.float32, shape=[None])
    action_one_hot = tf.one_hot(self.action_in, depth=action_size)
    
    self.q_state_local = self.build_model(action_size, "local")
    self.q_state_target = self.build_model(action_size, "target")
    
    self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state_local, action_one_hot), axis=1)
    self.loss = tf.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
    
    self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="local")
    self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")
    self.updater = tf.group([tf.assign(t, t + tau*(l-t)) for t,l in zip(self.target_vars, self.local_vars)])
    
  def build_model(self, action_size, scope):
    with tf.variable_scope(scope):
      hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
      q_state = tf.layers.dense(hidden1, action_size, activation=None)
      return q_state
      
  def update_model(self, session, state, action, q_target):
    feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
    session.run([self.optimizer, self.updater], feed_dict=feed)
      
  def get_q_state(self, session, state, use_target=False):
    q_state_op = self.q_state_target if use_target else self.q_state_local
    q_state = session.run(q_state_op, feed_dict={self.state_in: state})
    return q_state