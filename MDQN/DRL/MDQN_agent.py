import numpy as np
import torch
from copy import deepcopy as copy

from .networks import MultiHeadDuelingDQN
from heuristic.basic import Agent


class MDQN(Agent):
    '''myopic DQN agent'''
    def __init__(self, state_size:int, action_size:int, n_heads:int, lr:float, seed:int=0):
        self.state_size, self.action_size, self.n_heads = state_size, action_size, n_heads
        torch.manual_seed(seed)
        self.net = MultiHeadDuelingDQN(self.state_size, self.action_size, n_heads)
        self.target_net = copy(self.net)
        self.current_head = 0
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.h = torch.arange(self.n_heads, dtype=torch.float32).view(1, -1)
        self.h_next = self.h + 1
        self.rho = 1 / 2 ** torch.arange(self.n_heads, dtype=torch.float32).view(1, -1)
 
    def act(self, state:np.ndarray, epsilon:float=0.0, idx_head:int=None):
        idx_head = self.current_head if idx_head is None else idx_head
        if self.rng.random() < epsilon:
            return self.rng.choice(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).view(1, -1)
            self.net.eval()
            q = self.net(state_tensor)[0, idx_head, :]
            self.net.train()
            return torch.argmax(q).item()

    def train(self, transition:tuple):
        s, a, r, s_next = transition
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()
        with torch.no_grad():
            self.net.eval()
            q_next = torch.max(self.target_net(s_next), dim=-1)[0]
            self.net.train()
            q_next = torch.cat([torch.zeros((q_next.shape[0], 1)), q_next[:,:-1]], dim=-1)
            target = (r.view(-1,1) + self.h * q_next) / self.h_next
        a_idx = a.view(-1,1,1).expand(-1, self.n_heads, 1)
        q = self.net(s).gather(-1, a_idx).squeeze(-1)
        loss = torch.mean(self.rho * (q - target) ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def load_parameters(self, parameters:dict):
        self.current_head = parameters["head"]
        self.net.load_state_dict(parameters["weights"])

    def get_parameters(self):
        return {"head": self.current_head, "weights": copy(self.net.state_dict())}
        # return {"head": self.current_head, "weights": self.net.state_dict()}

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
