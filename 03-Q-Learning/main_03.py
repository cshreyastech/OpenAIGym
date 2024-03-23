import gymnasium as gym

from agent_class_03 import Agent
from qagent_03 import QAgent
import time

# from gym.envs.registration import register
from IPython.display import clear_output

if __name__ == '__main__':
  # env_name = "CartPole-v1"
  # env_name = "MountainCar-v0"
  # env_name = "MountainCarContinuous-v0"
  # env_name = "Acrobot-v1"
  # env_name = "Pendulum-v1"
  env_name = "FrozenLake-v1"
  env = gym.make(env_name, desc=None, map_name="4x4", is_slippery=False)
  num_games = 100

  agent = QAgent(env)
  

  total_reward = 0
  for ep in range(num_games):
    state, probability = env.reset()
    done = False
    while not done:
      action = agent.get_action(state)
      next_state, reward, done, truncated, info = env.step(action)
      agent.train((state,action,next_state,reward,done))
      state = next_state
      total_reward += reward
      # print("s:", state, "a:", action)
      print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, agent.eps))
      # env.render()
      # print(q_agent.q_table)
      # time.sleep(0.5)
      # clear_output(wait=True)