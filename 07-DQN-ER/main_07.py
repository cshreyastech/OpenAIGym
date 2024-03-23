import gymnasium as gym

from dqnagent_07 import DQNAgent
import time

# from gym.envs.registration import register
from IPython.display import clear_output

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if __name__ == '__main__':
  env_name = "CartPole-v1"
  # env_name = "MountainCar-v0"
  # env_name = "MountainCarContinuous-v0"
  # env_name = "Acrobot-v1"
  # env_name = "Pendulum-v1"
  # env_name = "FrozenLake-v1"
  env = gym.make(env_name)
  num_games = 100

  agent = DQNAgent(env)
  

  total_reward = 0
  for ep in range(num_games):
    state, probability = env.reset()
    done = False
    while not done:
      action = agent.get_action(state)
      next_state, reward, done, truncated, info = env.step(action)
      agent.train(state,action,next_state,reward,done)
      state = next_state
      total_reward += reward
      # print("s:", state, "a:", action)
      # print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))
      # env.render()
      # with tf.variable_scope("q_table", reuse=True):
      #   weights = agent.sess.run(tf.get_variable("kernel")) # (state_size, action_size)
        # print(weights)
      # time.sleep(0.1)
      # clear_output(wait=True)

  print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))