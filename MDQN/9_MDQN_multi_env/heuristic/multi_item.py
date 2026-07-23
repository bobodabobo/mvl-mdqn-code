from copy import deepcopy as copy

import numpy as np

from DRL.configs import DQN_config

from .basic import Agent, evaluate_policy_multi_process


def _build_transition_model(env, states, state_index, target_state=None):
    n_states = len(states)
    n_demands = len(env.joint_demand_support)
    next_indices = np.zeros((n_states, n_demands), dtype=np.int32)
    step_costs = np.full(n_states, np.inf, dtype=np.float64)
    feasible = np.zeros(n_states, dtype=bool)

    q1_max, q2_max = int(env.max_order_quantities[0]), int(env.max_order_quantities[1])
    low_1, low_2 = int(env.inventory_lows[0]), int(env.inventory_lows[1])
    high_1, high_2 = int(env.inventory_highs[0]), int(env.inventory_highs[1])
    order_cost_1, order_cost_2 = float(env.ordering_cost_rates[0]), float(env.ordering_cost_rates[1])
    holding_1, holding_2 = float(env.holding_cost_rates[0]), float(env.holding_cost_rates[1])
    shortage_1, shortage_2 = float(env.shortage_cost_rates[0]), float(env.shortage_cost_rates[1])
    fixed_order_cost = float(env.fixed_order_cost)

    for state_idx, (x1, x2) in enumerate(states):
        if target_state is None:
            q1, q2 = 0, 0
            feasible[state_idx] = True
        else:
            s1, s2 = int(target_state[0]), int(target_state[1])
            q1, q2 = s1 - x1, s2 - x2
            feasible[state_idx] = (
                q1 >= 0
                and q2 >= 0
                and q1 <= q1_max
                and q2 <= q2_max
                and (q1 + q2) > 0
            )
            if not feasible[state_idx]:
                continue

        timely_cost = order_cost_1 * q1 + order_cost_2 * q2
        if q1 + q2 > 0:
            timely_cost += fixed_order_cost

        inventory_1 = x1 + q1
        inventory_2 = x2 + q2
        delayed_expectation = 0.0
        for demand_idx, ((d1, d2), probability) in enumerate(env.joint_demand_support):
            next_1 = min(max(inventory_1 - d1, low_1), high_1)
            next_2 = min(max(inventory_2 - d2, low_2), high_2)
            delayed_expectation += probability * (
                holding_1 * max(next_1, 0)
                + holding_2 * max(next_2, 0)
                + shortage_1 * max(-next_1, 0)
                + shortage_2 * max(-next_2, 0)
            )
            next_indices[state_idx, demand_idx] = state_index[(next_1, next_2)]
        step_costs[state_idx] = timely_cost + delayed_expectation
    return feasible, step_costs, next_indices


def _compute_sigma_states(env, s1: int, s2: int, states, state_index, probabilities, no_order_model):
    feasible_order, order_step_costs, order_next_indices = _build_transition_model(
        env,
        states,
        state_index,
        target_state=(s1, s2),
    )
    _, no_order_step_costs, no_order_next_indices = no_order_model
    gamma = DQN_config["gamma"]
    values = np.zeros(len(states), dtype=np.float64)

    for _ in range(400):
        q_no_order = no_order_step_costs + gamma * np.sum(values[no_order_next_indices] * probabilities, axis=1)
        q_order = np.full_like(q_no_order, np.inf)
        if np.any(feasible_order):
            q_order[feasible_order] = order_step_costs[feasible_order] + gamma * np.sum(
                values[order_next_indices[feasible_order]] * probabilities,
                axis=1,
            )
        updated_values = np.minimum(q_no_order, q_order)
        if np.max(np.abs(updated_values - values)) < 1e-6:
            values = updated_values
            break
        values = updated_values

    q_no_order = no_order_step_costs + gamma * np.sum(values[no_order_next_indices] * probabilities, axis=1)
    q_order = np.full_like(q_no_order, np.inf)
    if np.any(feasible_order):
        q_order[feasible_order] = order_step_costs[feasible_order] + gamma * np.sum(
            values[order_next_indices[feasible_order]] * probabilities,
            axis=1,
        )
    return [
        (int(states[state_idx][0]), int(states[state_idx][1]))
        for state_idx in range(len(states))
        if feasible_order[state_idx] and q_order[state_idx] + 1e-10 < q_no_order[state_idx]
    ]


class SigmaSPolicy(Agent):
    def __init__(self, env, parameters: dict = None):
        self.env = copy(env)
        self.inventory_highs = self.env.inventory_highs.copy()
        self.max_order_quantities = self.env.max_order_quantities.copy()
        self.states = self.env.enumerate_states()
        self.state_index = {state: idx for idx, state in enumerate(self.states)}
        self.probabilities = np.array(
            [probability for _, probability in self.env.joint_demand_support],
            dtype=np.float64,
        ).reshape(1, -1)
        self.no_order_model = _build_transition_model(
            self.env,
            self.states,
            self.state_index,
            target_state=None,
        )
        if parameters is None:
            parameters = {
                "S1": 0,
                "S2": 0,
                "sigma_states": [],
                "sigma_size": 0,
            }
        self.set_parameters(parameters)

    def set_parameters(self, parameters: dict):
        super().set_parameters(parameters)
        self.sigma_state_set = {tuple(state) for state in self.sigma_states}

    def act(self, obs):
        state = (int(obs[0]), int(obs[1]))
        if state not in self.sigma_state_set:
            return 0
        q1 = int(np.clip(self.S1 - obs[0], 0, self.max_order_quantities[0]))
        q2 = int(np.clip(self.S2 - obs[1], 0, self.max_order_quantities[1]))
        return q1 * (self.max_order_quantities[1] + 1) + q2

    def train(self, length: int = None, repeats: int = 1):
        parameters_list = []
        for s1 in range(int(self.inventory_highs[0]) + 1):
            for s2 in range(int(self.inventory_highs[1]) + 1):
                sigma_states = _compute_sigma_states(
                    self.env,
                    s1,
                    s2,
                    self.states,
                    self.state_index,
                    self.probabilities,
                    self.no_order_model,
                )
                parameters_list.append(
                    {
                        "S1": s1,
                        "S2": s2,
                        "sigma_states": sigma_states,
                        "sigma_size": len(sigma_states),
                    }
                )
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log


class FixedOrderMI(Agent):
    def __init__(self, env, parameters: dict = None):
        self.env = copy(env)
        self.max_order_quantities = self.env.max_order_quantities.copy()
        if parameters is None:
            parameters = {"q1": 0, "q2": 0}
        self.set_parameters(parameters)

    def act(self, obs):
        del obs
        q1 = int(np.clip(self.q1, 0, self.max_order_quantities[0]))
        q2 = int(np.clip(self.q2, 0, self.max_order_quantities[1]))
        return q1 * (self.max_order_quantities[1] + 1) + q2

    def train(self, length: int = None, repeats: int = 1):
        parameters_list = []
        for q1 in range(int(self.max_order_quantities[0]) + 1):
            for q2 in range(int(self.max_order_quantities[1]) + 1):
                parameters_list.append({"q1": q1, "q2": q2})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log
