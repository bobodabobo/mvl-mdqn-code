import numpy as np
import torch
from copy import deepcopy as copy
from torch.distributions import Categorical

from heuristic.basic import Agent

from .networks import MultiHeadDiscreteActor, MultiHeadDuelingDQN


class MSAC(Agent):
    '''Multi-head discrete SAC with an MDQN-style critic.'''
    def __init__(self,
                 state_size:int,
                 action_size:int,
                 n_heads:int,
                 lr:float,
                 tau:float,
                 alpha_init:float,
                 alpha_lr:float,
                 target_entropy_scale:float,
                 lambda_anc:float=10.0 / 7.0,
                 seed:int=0):
        self.state_size, self.action_size, self.n_heads = state_size, action_size, n_heads
        self.tau = tau
        self.lambda_anc = lambda_anc
        self.current_head = 0
        torch.manual_seed(seed)
        self.actor = MultiHeadDiscreteActor(state_size, action_size, n_heads)
        self.critic = MultiHeadDuelingDQN(state_size, action_size, n_heads)
        self.target_critic = copy(self.critic)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy_scale * np.log(action_size)
        self.horizons = torch.arange(1, n_heads + 1, dtype=torch.float32).view(1, -1)
        self.prev_ratios = (self.horizons - 1.0) / self.horizons
        self.loss_weights = self._build_normalized_loss_weights(n_heads, lambda_anc)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @staticmethod
    def _build_normalized_loss_weights(n_heads:int, lambda_anc:float):
        weights = torch.ones(n_heads, dtype=torch.float32)
        if lambda_anc <= 0:
            return weights
        sap_weights = torch.zeros(n_heads, dtype=torch.float32)
        for idx_head in range(1, n_heads):
            sap_weights[:idx_head] += 1.0 / idx_head
        return weights + lambda_anc * sap_weights

    def act(self, state:np.ndarray, deterministic:bool=False, idx_head:int=None):
        idx_head = self.current_head if idx_head is None else idx_head
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).view(1, -1)
            probs, log_probs = self.actor.get_policy(state_tensor)
            probs = probs[0, idx_head]
            log_probs = log_probs[0, idx_head]
            if deterministic:
                action = torch.argmax(probs).view(1)
            else:
                action = Categorical(probs=probs).sample().view(1)
            log_prob = log_probs[action].flatten()
        return action.item(), log_prob.item()

    def train(self, batch:tuple, gamma:float):
        states, actions, rewards, states_next = batch
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        states_next = torch.as_tensor(states_next, dtype=torch.float32)

        with torch.no_grad():
            next_probs, next_log_probs = self.actor.get_policy(states_next)
            next_q = self.target_critic(states_next)
            next_v = (next_probs * (next_q - self.alpha.detach() * next_log_probs)).sum(dim=-1)
            horizons = self.horizons.to(rewards.device)
            prev_ratios = self.prev_ratios.to(rewards.device)
            target = rewards.view(-1, 1) / horizons
            if self.n_heads > 1:
                target[:, 1:] += gamma * prev_ratios[:, 1:] * next_v[:, :-1]

        action_idx = actions.view(-1, 1, 1).expand(-1, self.n_heads, 1)
        critic_values = self.critic(states).gather(-1, action_idx).squeeze(-1)
        head_losses = 0.5 * ((critic_values - target) ** 2).mean(dim=0)
        critic_loss = torch.sum(self.loss_weights.to(critic_values.device) * head_losses)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        probs, log_probs = self.actor.get_policy(states)
        q_pi = self.critic(states).detach()
        actor_head_losses = (probs * (self.alpha.detach() * log_probs - q_pi)).sum(dim=-1).mean(dim=0)
        actor_loss = actor_head_losses.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return float((critic_loss + actor_loss).item())

    def update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def load_parameters(self, parameters:dict):
        self.current_head = parameters["head"]
        self.actor.load_state_dict(parameters["actor"])
        self.critic.load_state_dict(parameters["critic"])
        self.target_critic.load_state_dict(parameters["target_critic"])
        self.log_alpha = torch.tensor(parameters["log_alpha"], dtype=torch.float32, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=self.alpha_opt.param_groups[0]["lr"])

    def get_parameters(self):
        return {
            "head": self.current_head,
            "actor": copy(self.actor.state_dict()),
            "critic": copy(self.critic.state_dict()),
            "target_critic": copy(self.target_critic.state_dict()),
            "log_alpha": float(self.log_alpha.detach().item()),
        }
