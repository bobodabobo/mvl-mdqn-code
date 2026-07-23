import numpy as np
import torch
from copy import deepcopy as copy

from .networks import MultiHeadDuelingDQN
from heuristic.basic import Agent


class MDQN(Agent):
    '''Myopic DQN agent with discounted targets and simplified SAP-inspired joint loss.'''
    def __init__(self,
                 state_size:int,
                 action_size:int,
                 n_heads:int,
                 lr:float,
                 lambda_anc:float=0.1,
                 seed:int=0):
        self.state_size, self.action_size, self.n_heads = state_size, action_size, n_heads
        self.lambda_anc = lambda_anc
        torch.manual_seed(seed)
        self.net = MultiHeadDuelingDQN(self.state_size, self.action_size, n_heads)
        self.target_net = copy(self.net)
        self.current_head = 0
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.horizons = torch.arange(1, self.n_heads + 1, dtype=torch.float32).view(1, -1)
        self.prev_ratios = (self.horizons - 1.0) / self.horizons
        self.loss_weights = self._build_normalized_loss_weights(self.n_heads, self.lambda_anc)

    @staticmethod
    def _build_normalized_loss_weights(n_heads:int, lambda_anc:float):
        # We approximate SAP with per-head coefficients: each head keeps its own
        # regression loss and receives extra weight from later horizons that rely on it.
        weights = torch.ones(n_heads, dtype=torch.float32)
        if lambda_anc <= 0:
            return weights
        sap_weights = torch.zeros(n_heads, dtype=torch.float32)
        for idx_head in range(1, n_heads):
            sap_weights[:idx_head] += 1.0 / idx_head
        return weights + lambda_anc * sap_weights
 
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

    def train(self, transition:tuple, gamma:float):
        s, a, r, s_next = transition
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()

        with torch.no_grad():
            self.target_net.eval()
            q_next = self.target_net(s_next)

            horizons = self.horizons.to(r.device)
            prev_ratios = self.prev_ratios.to(r.device)
            target = r.view(-1, 1) / horizons
            if self.n_heads > 1:
                q_next_prev = torch.max(q_next[:, :-1, :], dim=-1)[0]
                target[:, 1:] += gamma * prev_ratios[:, 1:] * q_next_prev

        a_idx = a.view(-1,1,1).expand(-1, self.n_heads, 1)
        q = self.net(s).gather(-1, a_idx).squeeze(-1)
        head_losses = 0.5 * ((q - target) ** 2).mean(dim=0)
        loss = torch.sum(self.loss_weights.to(q.device) * head_losses)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def load_parameters(self, parameters:dict):
        self.current_head = parameters["head"]
        self.net.load_state_dict(parameters["weights"])
        self.update_target_net()

    def get_parameters(self):
        return {"head": self.current_head, "weights": copy(self.net.state_dict())}
        # return {"head": self.current_head, "weights": self.net.state_dict()}

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
