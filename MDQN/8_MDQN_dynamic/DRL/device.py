from __future__ import annotations

import torch


CPU_DEVICE = torch.device("cpu")


def resolve_cpu_device(device: str | torch.device | None = None) -> torch.device:
    """Return the normalized CPU device and reject accelerator devices."""
    resolved = CPU_DEVICE if device is None else torch.device(device)
    if resolved.type != "cpu":
        raise ValueError(f"This project is configured for CPU execution only, got device={resolved}.")
    return CPU_DEVICE
