import torch
import torch.nn as nn


HIDDEN_SIZE = 64


class _Head(nn.Module):
    '''Prediction head with dueling architecture'''
    def __init__(self, input_size:int, output_size:int):
        super(_Head, self).__init__()
        self.value = nn.Linear(input_size, 1)
        self.advantage = nn.Linear(input_size, output_size)
    def forward(self, x):
        value = self.value(x)
        advantage = self.advantage(x)
        x = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return x


class MultiHeadDuelingDQN(nn.Module):
    '''Multi-head dueling DQN'''
    def __init__(self, input_size:int, output_size:int, num_heads:int):
        super(MultiHeadDuelingDQN, self).__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh()
        )
        self.heads = nn.ModuleList([_Head(HIDDEN_SIZE, output_size) for _ in range(num_heads)])
    def forward(self, x):
        x = self.main_branch(x)
        return torch.stack([head(x) for head in self.heads], dim=1)


class DuelingDQN(nn.Module):
    '''Dueling DQN'''
    def __init__(self, input_size:int, output_size:int):
        super(DuelingDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh()
        )
        self.head = _Head(HIDDEN_SIZE, output_size)
    def forward(self, x):
        x = self.layers(x)
        x = self.head(x)
        return x
