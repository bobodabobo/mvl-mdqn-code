import numpy as np
import torch
from copy import deepcopy as copy

from .device import resolve_cpu_device
from .networks import DuelingDQN
from heuristic.basic import Agent


class DQN(Agent):
    '''basic DQN agent'''
    def __init__(self, state_size:int, action_size:int, lr:float, seed:int=0, device:str="cpu"):
        self.state_size, self.action_size = state_size, action_size
        self.device = resolve_cpu_device(device)
        torch.manual_seed(seed)
        self.net = DuelingDQN(self.state_size, self.action_size).to(self.device)
        self.loss_f = torch.nn.MSELoss()
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
 
    def act(self, state:np.ndarray, epsilon:float=0.0):
        if self.rng.random() < epsilon:
            action = self.rng.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
                self.net.eval()
                q = self.net(state)
                self.net.train()
                action = torch.argmax(q, dim=1)[0].item()
        return action

    def train(self, transition:tuple, gamma:float):
        s, a, r, s_next = transition
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.net.eval()
            q_next = torch.max(self.net(s_next), dim=-1)[0]
            self.net.train()
            target = r + gamma * q_next
        q = self.net(s).gather(1, a.view(-1, 1)).flatten()
        loss = self.loss_f(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        loss_to_show = loss.item()
        return loss_to_show

    def load_parameters(self, parameters:dict):
        self.net.load_state_dict(parameters)

    def get_parameters(self):
        return copy(self.net.state_dict())


class TDQN(DQN):
    '''target DQN agent'''
    def __init__(self, state_size:int, action_size:int, lr:float, seed:int=0, device:str="cpu"):
        super().__init__(state_size, action_size, lr, seed, device)
        self.target_net = DuelingDQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
    
    def train(self, transition:tuple, gamma:float):
        s, a, r, s_ = transition
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s_ = torch.as_tensor(s_, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_next = torch.max(self.target_net(s_), dim=-1)[0]
            target = r + gamma * q_next
        q = self.net(s).gather(1, a.view(-1, 1)).flatten()
        loss = self.loss_f(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        loss_to_show = loss.item()
        return loss_to_show

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())


class DDQN(TDQN):
    '''double DQN agent'''
    def __init__(self, state_size:int, action_size:int, lr:float, seed:int=0, device:str="cpu"):
        super().__init__(state_size, action_size, lr, seed, device)
    
    def train(self, transition:tuple, gamma:float):
        s, a, r, s_next = transition
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.net.eval()
            actions = self.net(s_next).argmax(dim=-1)
            q_next = self.target_net(s_next).gather(1, actions.view(-1, 1)).flatten()
            self.net.train()
            target = r + gamma * q_next
        q = self.net(s).gather(1, a.view(-1, 1)).flatten()
        loss = self.loss_f(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        loss_to_show = loss.item()
        return loss_to_show
