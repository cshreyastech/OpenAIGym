import gymnasium as gym
import numpy as np

from ddqnagent_08 import DoubleDQNAgent
import time

# from gym.envs.registration import register
from IPython.display import clear_output

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt

if __name__ == '__main__':
  env_name = "CartPole-v1"
  # env_name = "MountainCar-v0"
  # env_name = "MountainCarContinuous-v0"
  # env_name = "Acrobot-v1"
  # env_name = "Pendulum-v1"
  # env_name = "FrozenLake-v1"
  env = gym.make(env_name)
  # num_games = 10

  # agent = DoubleDQNAgent(env)
  

  # num_runs = 10
  # run_rewards = []


  # for n in range(num_runs):
  #   print("Run {}".format(n))
  #   ep_rewards = []
  #   agent = None
  #   agent = DoubleDQNAgent(env)
  #   num_episodes = 200



  # total_reward = 0
  # for ep in range(num_games):
  #   state, probability = env.reset()
  #   done = False
  #   while not done:
  #     action = agent.get_action(state)
  #     next_state, reward, done, truncated, info = env.step(action)
  #     agent.train(state,action,next_state,reward,done)
  #     state = next_state
  #     total_reward += reward
  #     # print("s:", state, "a:", action)
  #     # print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))
  #     # env.render()
  #     # with tf.variable_scope("q_table", reuse=True):
  #     #   weights = agent.sess.run(tf.get_variable("kernel")) # (state_size, action_size)
  #       # print(weights)
  #     # time.sleep(0.1)
  #     # clear_output(wait=True)

  # print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))


  num_runs = 4
  run_rewards = []

  for n in range(num_runs):
    print("Run {}".format(n))
    ep_rewards = []
    agent = None
    agent = DoubleDQNAgent(env)
    num_episodes = 200

    for ep in range(num_episodes):
      state, probability = env.reset()
      total_reward = 0
      done = False
      while not done:
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.train(state, action, next_state, reward, done, use_DDQN=(n%2==0))
        #env.render()
        total_reward += reward
        state = next_state

      ep_rewards.append(total_reward)
      #print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))
        
    run_rewards.append(ep_rewards)

  for n, ep_rewards in enumerate(run_rewards):
    x = range(len(ep_rewards))
    cumsum = np.cumsum(ep_rewards)
    avgs = [cumsum[ep]/(ep+1) if ep<100 else (cumsum[ep]-cumsum[ep-100])/100 for ep in x]
    col = "r" if (n%2==0) else "b"
    plt.plot(x, avgs, color=col, label=n)
    
  plt.title("DDQN vs DQN performance")
  plt.xlabel("Episode")
  plt.ylabel("Last 100 episode average rewards")
  plt.legend()
  plt.show()
     