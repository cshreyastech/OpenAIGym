import gymnasium as gym

from agent_class_06 import HillClimbingAgent
import time


if __name__ == '__main__':
  env_name = "CartPole-v1"
  # env_name = "MountainCar-v0"
  # env_name = "MountainCarContinuous-v0"
  # env_name = "Acrobot-v1"
  # env_name = "Pendulum-v1"
  # env_name = "FrozenLake-v1"
  env = gym.make(env_name)
  num_games = 100

  agent = HillClimbingAgent(env)
  

  total_reward = 0
  for ep in range(num_games):
    state, probability = env.reset()
    done = False
    while not done:
      action = agent.get_action(state)
      next_state, reward, done, truncated, info = env.step(action)
      
      # state = next_state
      total_reward += reward
      # print("s:", state, "a:", action)
      # env.render()
      # time.sleep(0.1)
      # clear_output(wait=True)
    agent.update_model(total_reward)
    print("Episode: {}, Total reward: {}".format(ep, total_reward))