import numpy as np
import torch
from copy import deepcopy as copy
from torch.distributions import Categorical

from .networks import DiscreteActor, DiscreteQCritic


class SAC:
    '''Discrete-action SAC agent.'''
    def __init__(self,
                 state_size:int,
                 action_size:int,
                 lr:float,
                 tau:float,
                 alpha_init:float,
                 alpha_lr:float,
                 target_entropy_scale:float,
                 seed:int=0):
        self.state_size, self.action_size = state_size, action_size
        self.tau = tau
        torch.manual_seed(seed)
        self.actor = DiscreteActor(state_size, action_size)
        self.q1 = DiscreteQCritic(state_size, action_size)
        self.q2 = DiscreteQCritic(state_size, action_size)
        self.q1_target = DiscreteQCritic(state_size, action_size)
        self.q2_target = DiscreteQCritic(state_size, action_size)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.AdamW(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.AdamW(self.q2.parameters(), lr=lr)
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy_scale * np.log(action_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state:np.ndarray, deterministic:bool=False):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).view(1, -1)
            probs, log_probs = self.actor.get_policy(state_tensor)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = Categorical(probs=probs).sample()
            log_prob = log_probs.gather(1, action.view(-1, 1)).flatten()
        return action.item(), log_prob.item()

    def train(self, batch:tuple, gamma:float):
        states, actions, rewards, states_next, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        states_next = torch.as_tensor(states_next, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)
        with torch.no_grad():
            next_probs, next_log_probs = self.actor.get_policy(states_next)
            next_q1 = self.q1_target(states_next)
            next_q2 = self.q2_target(states_next)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=-1)
            target_q = rewards + gamma * (1.0 - dones) * next_v
        current_q1 = self.q1(states).gather(1, actions.view(-1, 1)).flatten()
        current_q2 = self.q2(states).gather(1, actions.view(-1, 1)).flatten()
        q1_loss = torch.nn.functional.mse_loss(current_q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(current_q2, target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()
        probs, log_probs = self.actor.get_policy(states)
        q1_pi = self.q1(states)
        q2_pi = self.q2(states)
        q_pi = torch.min(q1_pi, q2_pi).detach()
        actor_loss = (probs * (self.alpha.detach() * log_probs - q_pi)).sum(dim=-1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return float((q1_loss + q2_loss + actor_loss).item())

    def update_target(self):
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def load_parameters(self, parameters:dict):
        self.actor.load_state_dict(parameters["actor"])
        self.q1.load_state_dict(parameters["q1"])
        self.q2.load_state_dict(parameters["q2"])
        self.q1_target.load_state_dict(parameters["q1_target"])
        self.q2_target.load_state_dict(parameters["q2_target"])
        self.log_alpha = torch.tensor(parameters["log_alpha"], dtype=torch.float32, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=self.alpha_opt.param_groups[0]["lr"])

    def get_parameters(self):
        return {
            "actor": copy(self.actor.state_dict()),
            "q1": copy(self.q1.state_dict()),
            "q2": copy(self.q2.state_dict()),
            "q1_target": copy(self.q1_target.state_dict()),
            "q2_target": copy(self.q2_target.state_dict()),
            "log_alpha": float(self.log_alpha.detach().item())
        }
