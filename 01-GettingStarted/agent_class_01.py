import random

class Agent():
  def __init__(self, env):
    self.action_space = env.action_space.n
    print("self.action_space", self.action_space)

  def get_action(self, state):
    # action = random.choice(range(self.action_space))
    pole_angle = state[2]
    action = 0 if pole_angle < 0 else 1
    return action