
from brain import Brain

class Agent:
    def __init__(self, n_actions):
        self.brain = Brain(n_actions)

    def update(self):
        self.brain.replay()

    def get_action(self, state, steps, mode='train'):
        """
        :param steps: 用来更新eposilon的步数, 可以是episode
        :type steps: int
        """
        action = self.brain.decide_action(state, steps, mode)
        return action
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target(self):
        self.brain.update_target_network()
    
    def save_model(self, filename):
        self.brain.save_net_state(filename)
    
    def load_model(self, filename):
        self.brain.load_net_state(filename)