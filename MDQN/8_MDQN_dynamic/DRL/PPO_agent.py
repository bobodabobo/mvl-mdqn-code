import numpy as np
import torch
from copy import deepcopy as copy
from torch.distributions import Categorical

from .device import resolve_cpu_device
from .networks import DiscreteActor, ValueCritic


class PPO:
    '''Discrete-action PPO agent.'''
    def __init__(self,
                 state_size:int,
                 action_size:int,
                 lr:float,
                 batch_size:int,
                 clip_ratio:float,
                 update_epochs:int,
                 value_coef:float,
                 entropy_coef:float,
                 max_grad_norm:float,
                 seed:int=0,
                 device:str="cpu"):
        self.state_size, self.action_size = state_size, action_size
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = resolve_cpu_device(device)
        torch.manual_seed(seed)
        self.actor = DiscreteActor(state_size, action_size).to(self.device)
        self.critic = ValueCritic(state_size).to(self.device)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

    def act(self, state:np.ndarray, deterministic:bool=False):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            logits = self.actor(state_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
        return action.item(), log_prob.item(), value.item()

    def evaluate_value(self, state:np.ndarray):
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            value = self.critic(state_tensor)
        return value.item()

    def train(self, batch:tuple):
        states, actions, old_log_probs, returns, advantages = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        indices = np.arange(states.shape[0])
        losses = []
        for _ in range(self.update_epochs):
            self.rng.shuffle(indices)
            for start in range(0, states.shape[0], self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                states_mb = states[batch_idx]
                actions_mb = actions[batch_idx]
                old_log_probs_mb = old_log_probs[batch_idx]
                returns_mb = returns[batch_idx]
                advantages_mb = advantages[batch_idx]
                logits = self.actor(states_mb)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()
                values = self.critic(states_mb)
                ratios = torch.exp(new_log_probs - old_log_probs_mb)
                surrogate_1 = ratios * advantages_mb
                surrogate_2 = torch.clamp(ratios,
                                          1.0 - self.clip_ratio,
                                          1.0 + self.clip_ratio) * advantages_mb
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                critic_loss = torch.nn.functional.mse_loss(values, returns_mb)
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()
                losses.append(loss.item())
        if not losses:
            return 0.0
        return float(np.mean(losses))

    def load_parameters(self, parameters:dict):
        self.actor.load_state_dict(parameters["actor"])
        self.critic.load_state_dict(parameters["critic"])

    def get_parameters(self):
        return {
            "actor": copy(self.actor.state_dict()),
            "critic": copy(self.critic.state_dict())
        }
