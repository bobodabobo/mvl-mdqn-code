import numpy as np


DYNAMIC_DEMAND_CENTER_MIN = 0.0
DYNAMIC_DEMAND_CENTER_MAX = 5.0
DYNAMIC_DEMAND_HALF_WIDTH = 2.0
DYNAMIC_DEMAND_LONG_RUN_MEAN = 0.5 * (DYNAMIC_DEMAND_CENTER_MIN + DYNAMIC_DEMAND_CENTER_MAX)
DYNAMIC_DEMAND_MAX = DYNAMIC_DEMAND_CENTER_MAX + DYNAMIC_DEMAND_HALF_WIDTH


def get_dynamic_demand_metadata(effective_lead_time: int) -> dict:
    period = 4 * effective_lead_time
    history_len = 2 * effective_lead_time
    return {
        "center_min": DYNAMIC_DEMAND_CENTER_MIN,
        "center_max": DYNAMIC_DEMAND_CENTER_MAX,
        "half_width": DYNAMIC_DEMAND_HALF_WIDTH,
        "long_run_mean": DYNAMIC_DEMAND_LONG_RUN_MEAN,
        "max_demand": DYNAMIC_DEMAND_MAX,
        "period": period,
        "history_len": history_len,
    }


class DynamicDemandGenerator:
    """Shared periodic dynamic-demand generator."""

    def __init__(self, effective_lead_time: int, seed: int | None = None):
        metadata = get_dynamic_demand_metadata(effective_lead_time)
        self.effective_lead_time = effective_lead_time
        self.period = metadata["period"]
        self.history_len = metadata["history_len"]
        self.reset_seed(seed)

    def reset_seed(self, seed: int | None):
        self.rng = np.random.default_rng(seed=seed)

    def demand_center(self, steps: np.ndarray, phase_offset: int) -> np.ndarray:
        angle = 2.0 * np.pi * (steps + phase_offset) / self.period
        return DYNAMIC_DEMAND_LONG_RUN_MEAN + DYNAMIC_DEMAND_LONG_RUN_MEAN * np.sin(angle)

    def sample_episode(self, max_steps: int, phase_offset: int) -> dict:
        steps = np.arange(-self.history_len, max_steps, dtype=np.int64)
        centers = self.demand_center(steps, phase_offset)
        demand_low = np.maximum(DYNAMIC_DEMAND_CENTER_MIN, centers - DYNAMIC_DEMAND_HALF_WIDTH)
        demand_high = centers + DYNAMIC_DEMAND_HALF_WIDTH
        demands = self.rng.uniform(demand_low, demand_high).astype(np.float32)
        return {
            "history": demands[: self.history_len],
            "demand_list": demands[self.history_len :],
            "demand_centers": centers[self.history_len :].astype(np.float32),
        }
