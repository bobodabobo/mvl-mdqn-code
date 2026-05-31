import numpy as np


class Agent:
    def __init__(self, distribution:np.ndarray):
        self.rng = np.random.default_rng(0)
        self.distribution = distribution
        n_states, n_actions = distribution.shape
        self.states = np.arange(n_states)
        self.actions = np.arange(n_actions)

    def act(self, state):
        if state not in self.states:
            raise ValueError(f"Invalid state: {state}")
        action = self.rng.choice(2, p=self.distribution[state, :])
        return action
    
    def get_prob(self, state, action):
        return self.distribution[state, action]


def cal_importance(explore_agent:Agent, target_agent:Agent, state, action):
    return target_agent.get_prob(state, action) / explore_agent.get_prob(state, action)


class ExploreAgent(Agent):
    def __init__(self):
        distribution = np.array((6/7, 1/7))[np.newaxis, :]
        distribution = np.repeat(distribution, 7, axis=0)
        super().__init__(distribution)


class TargetAgent(Agent):
    def __init__(self):
        distribution = np.zeros((7, 2), dtype=np.float64)
        distribution[:, -1] += 1
        super().__init__(distribution)
