import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


multi_item_configs = [
    {
        "demand_means": [2, 3],
        "demand_dist": "poisson_truncated",
        "demand_max": [5, 6],
        "seed": 0,
        "fixed_order_cost": 4.0,
        "ordering_cost_rates": [0.4, 0.6],
        "holding_cost_rates": [1.0, 1.2],
        "shortage_cost_rates": [6.0, 7.0],
        "inventory_lows": [-4, -4],
        "inventory_highs": [6, 6],
        "max_order_quantities": [8, 8],
        "max_steps": 1000,
    },
    {
        "demand_means": [1, 3],
        "demand_dist": "poisson_truncated",
        "demand_max": [5, 6],
        "seed": 0,
        "fixed_order_cost": 6.0,
        "ordering_cost_rates": [0.4, 0.8],
        "holding_cost_rates": [1.0, 1.5],
        "shortage_cost_rates": [7.0, 9.0],
        "inventory_lows": [-4, -4],
        "inventory_highs": [6, 6],
        "max_order_quantities": [8, 8],
        "max_steps": 1000,
    },
]


def _truncated_poisson_pmf(mean: float, demand_max: int):
    probs = np.zeros(demand_max + 1, dtype=np.float64)
    probs[0] = math.exp(-mean)
    for idx in range(1, demand_max):
        probs[idx] = probs[idx - 1] * mean / idx
    probs[demand_max] = max(0.0, 1.0 - np.sum(probs[:-1]))
    return probs


class DemandGenerator:
    """Independent truncated Poisson demand for a two-item joint replenishment system."""

    def __init__(self, means, demand_max, seed: int = 0, demand_dist: str = "poisson_truncated"):
        if demand_dist != "poisson_truncated":
            raise ValueError(f"Unsupported demand distribution: {demand_dist}.")
        self.means = np.array(means, dtype=np.float64)
        self.demand_max = np.array(demand_max, dtype=np.int64)
        self.demand_dist = demand_dist
        self.reset_seed(seed)

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, size: int = 1):
        draws = self.rng.poisson(self.means, size=(size, len(self.means)))
        return np.minimum(draws, self.demand_max).astype(np.int64)


class MultiItemInventory(gym.Env):
    """Two-item periodic-review system with joint fixed ordering cost and zero lead time."""

    def __init__(self, config: dict):
        super().__init__()
        self.demand_means = np.array(config["demand_means"], dtype=np.float64)
        self.demand_max = np.array(config["demand_max"], dtype=np.int64)
        self.fixed_order_cost = float(config["fixed_order_cost"])
        self.ordering_cost_rates = np.array(config["ordering_cost_rates"], dtype=np.float64)
        self.holding_cost_rates = np.array(config["holding_cost_rates"], dtype=np.float64)
        self.shortage_cost_rates = np.array(config["shortage_cost_rates"], dtype=np.float64)
        self.inventory_lows = np.array(config["inventory_lows"], dtype=np.int64)
        self.inventory_highs = np.array(config["inventory_highs"], dtype=np.int64)
        self.max_order_quantities = np.array(config["max_order_quantities"], dtype=np.int64)
        self.max_steps = config["max_steps"]
        self.reward_delay_time = 0
        self.n_items = 2
        self.demand_generator = DemandGenerator(
            means=config["demand_means"],
            demand_max=config["demand_max"],
            seed=config["seed"],
            demand_dist=config["demand_dist"],
        )
        self.demand_pmfs = [
            _truncated_poisson_pmf(mean, dmax)
            for mean, dmax in zip(self.demand_means, self.demand_max)
        ]
        self.joint_demand_support = self._build_joint_demand_support()
        action_size = int(np.prod(self.max_order_quantities + 1))
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(
            low=self.inventory_lows.astype(np.int64),
            high=self.inventory_highs.astype(np.int64),
            dtype=np.int64,
        )
        self._set_seed(config["seed"])

    def _build_joint_demand_support(self):
        support = []
        for demand_1, prob_1 in enumerate(self.demand_pmfs[0]):
            for demand_2, prob_2 in enumerate(self.demand_pmfs[1]):
                support.append(((demand_1, demand_2), float(prob_1 * prob_2)))
        return support

    def _set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.action_space.seed(seed)
        self.demand_generator.reset_seed(seed)

    def _initialize(self):
        self.inventory_last = np.zeros(self.n_items, dtype=np.int64)
        self.inventory = np.zeros(self.n_items, dtype=np.int64)
        self.current_step = 0
        self.timely_cost = 0.0
        self.delayed_cost = 0.0
        self.variable_order_cost = 0.0
        self.holding_cost_total = 0.0
        self.shortage_cost_total = 0.0
        self.total_cost = 0.0
        self.action = 0
        self.order_quantities = np.zeros(self.n_items, dtype=np.int64)
        self.demand = np.zeros(self.n_items, dtype=np.int64)

    def _decode_action(self, action: int):
        q2 = action % (self.max_order_quantities[1] + 1)
        q1 = action // (self.max_order_quantities[1] + 1)
        return np.array([q1, q2], dtype=np.int64)

    def _clip_inventory(self):
        clipped = np.clip(self.inventory, self.inventory_lows, self.inventory_highs)
        truncated = bool(np.any(clipped != self.inventory))
        self.inventory = clipped.astype(np.int64)
        return truncated

    def _get_obs(self):
        return self.inventory.astype(np.int64).copy()

    def _get_info(self):
        return {
            "step": self.current_step,
            "costs": {
                "variable_order": self.variable_order_cost,
                "holding": self.holding_cost_total,
                "shortage": self.shortage_cost_total,
                "timely": self.timely_cost,
                "delayed": self.delayed_cost,
                "total": self.total_cost,
            },
            "inventory_last": self.inventory_last.tolist(),
            "inventory": self.inventory.tolist(),
            "order_quantities": self.order_quantities.tolist(),
            "demand": self.demand.tolist(),
            "action": self.action,
        }

    def enumerate_states(self):
        return [
            (inv_1, inv_2)
            for inv_1 in range(int(self.inventory_lows[0]), int(self.inventory_highs[0]) + 1)
            for inv_2 in range(int(self.inventory_lows[1]), int(self.inventory_highs[1]) + 1)
        ]

    def clip_state(self, state):
        state_arr = np.array(state, dtype=np.int64)
        clipped = np.clip(state_arr, self.inventory_lows, self.inventory_highs)
        return clipped.astype(np.int64)

    def reset(self, seed: int = None):
        if seed is not None:
            super().reset(seed=seed)
            self._set_seed(seed)
        else:
            super().reset()
        self._initialize()
        self.demand_list = self.demand_generator.sample(self.max_steps)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.inventory_last = self.inventory.copy()

        action = int(action)
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}.")

        self.action = action
        self.order_quantities = self._decode_action(action)
        if np.any(self.order_quantities > self.max_order_quantities):
            raise ValueError(f"Invalid order quantities {self.order_quantities}.")

        self.timely_cost = self.fixed_order_cost if np.sum(self.order_quantities) > 0 else 0.0
        self.variable_order_cost = float(np.dot(self.ordering_cost_rates, self.order_quantities))
        self.timely_cost += self.variable_order_cost

        self.inventory = self.inventory + self.order_quantities
        truncated = self._clip_inventory()

        self.demand = self.demand_list[self.current_step].astype(np.int64)
        self.inventory = self.inventory - self.demand
        truncated = self._clip_inventory() or truncated

        positive_inventory = np.maximum(self.inventory, 0)
        backlog = np.maximum(-self.inventory, 0)
        self.holding_cost_total = float(np.dot(self.holding_cost_rates, positive_inventory))
        self.shortage_cost_total = float(np.dot(self.shortage_cost_rates, backlog))
        self.delayed_cost = self.holding_cost_total + self.shortage_cost_total
        self.total_cost = self.timely_cost + self.delayed_cost
        reward = -self.total_cost

        observation = self._get_obs()
        info = self._get_info()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, info

    def cal_action_distance(self, action_0, action_1):
        q0 = self._decode_action(int(action_0))
        q1 = self._decode_action(int(action_1))
        return int(np.abs(q0 - q1).sum())

    def close(self):
        pass
