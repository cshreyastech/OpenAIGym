import gymnasium as gym
from agent_class_01 import Agent


if __name__ == '__main__':
  env_name = 'CartPole-v1'
  env = gym.make(env_name)
  num_games = 10

  agent = Agent(env)

  state, probability = env.reset()

  for _ in range(200):
    # action = env.action_space.sample()
    action = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action)
    # env.render()