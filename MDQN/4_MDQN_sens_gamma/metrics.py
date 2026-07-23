import numpy as np


def _extract_cost_components(frame):
    if isinstance(frame, dict):
        return (
            float(frame["timely"]),
            float(frame["delayed"]),
            float(frame["total"]),
        )
    if len(frame) != 3:
        raise ValueError(f"Expected (timely, delayed, total), got {frame!r}")
    timely, delayed, total = frame
    return float(timely), float(delayed), float(total)


def _discounted_average(values: np.ndarray, gamma: float):
    discounts = np.power(gamma, np.arange(values.size, dtype=np.float64))
    return float(np.dot(discounts, values) / np.sum(discounts))


def evaluate_cost_sequence(cost_frames, gamma: float, delay_time: int = 0, use_delayed_cost_assignment: bool = True):
    timely_costs = []
    delayed_costs = []
    total_costs = []
    for frame in cost_frames:
        timely, delayed, total = _extract_cost_components(frame)
        timely_costs.append(timely)
        delayed_costs.append(delayed)
        total_costs.append(total)

    total_costs = np.asarray(total_costs, dtype=np.float64)
    if total_costs.size == 0:
        raise ValueError("cost_frames must contain at least one element.")

    mean_cost = float(np.mean(total_costs))
    raw_discounted_cost = mean_cost if np.isclose(gamma, 1.0) else _discounted_average(total_costs, gamma)

    # For gamma<1, evaluate discounted performance on the same delayed-cost-assigned
    # sequence used by MDQN. This avoids overweighting uncontrollable early delayed
    # costs in lead-time systems.
    if use_delayed_cost_assignment and delay_time > 0 and not np.isclose(gamma, 1.0):
        timely_costs = np.asarray(timely_costs, dtype=np.float64)
        delayed_costs = np.asarray(delayed_costs, dtype=np.float64)
        if total_costs.size > delay_time:
            assigned_costs = timely_costs[:-delay_time] + (gamma ** delay_time) * delayed_costs[delay_time:]
            discounted_cost = _discounted_average(assigned_costs, gamma)
            mean_assigned_cost = float(np.mean(assigned_costs))
            n_assigned_steps = int(assigned_costs.size)
        else:
            assigned_costs = np.empty((0,), dtype=np.float64)
            discounted_cost = raw_discounted_cost
            mean_assigned_cost = mean_cost
            n_assigned_steps = 0
    else:
        assigned_costs = total_costs
        discounted_cost = raw_discounted_cost
        mean_assigned_cost = mean_cost
        n_assigned_steps = int(total_costs.size)

    return {
        "discounted_cost": discounted_cost,
        "mean_cost": mean_cost,
        "mean_assigned_cost": mean_assigned_cost,
        "raw_discounted_cost": raw_discounted_cost,
        "n_steps": int(total_costs.size),
        "n_assigned_steps": n_assigned_steps,
    }


def aggregate_episode_metrics(episode_costs, gamma: float, delay_time: int = 0, use_delayed_cost_assignment: bool = True):
    metrics = [
        evaluate_cost_sequence(
            costs,
            gamma,
            delay_time=delay_time,
            use_delayed_cost_assignment=use_delayed_cost_assignment,
        )
        for costs in episode_costs
    ]
    return {
        "performance": float(np.mean([item["discounted_cost"] for item in metrics])),
        "mean_cost": float(np.mean([item["mean_cost"] for item in metrics])),
        "mean_assigned_cost": float(np.mean([item["mean_assigned_cost"] for item in metrics])),
        "raw_discounted_cost": float(np.mean([item["raw_discounted_cost"] for item in metrics])),
        "n_episodes": len(metrics),
        "delay_time": int(delay_time),
        "use_delayed_cost_assignment": bool(use_delayed_cost_assignment),
    }
