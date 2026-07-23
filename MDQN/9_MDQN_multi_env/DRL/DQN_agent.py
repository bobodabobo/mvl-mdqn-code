from copy import deepcopy as copy

import numpy as np
import torch

from heuristic.basic import Agent

from .networks import DuelingDQN


class DDQN(Agent):
    """Double DQN agent used as the only value-based baseline in this workspace."""

    def __init__(self, state_size: int, action_size: int, lr: float, seed: int = 0):
        self.state_size = state_size
        self.action_size = action_size
        torch.manual_seed(seed)
        self.net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.loss_f = torch.nn.MSELoss()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, epsilon: float = 0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_size))
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).view(1, -1)
            self.net.eval()
            q_values = self.net(state_tensor)
            self.net.train()
            return int(torch.argmax(q_values, dim=1)[0].item())

    def train(self, transition: tuple, gamma: float):
        states, actions, rewards, next_states = transition
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        with torch.no_grad():
            self.net.eval()
            next_actions = self.net(next_states).argmax(dim=-1)
            next_q = self.target_net(next_states).gather(1, next_actions.view(-1, 1)).flatten()
            self.net.train()
            target = rewards + gamma * next_q
        q_values = self.net(states).gather(1, actions.view(-1, 1)).flatten()
        loss = self.loss_f(q_values, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def load_parameters(self, parameters: dict):
        self.net.load_state_dict(parameters)
        self.target_net.load_state_dict(parameters)

    def get_parameters(self):
        return copy(self.net.state_dict())
