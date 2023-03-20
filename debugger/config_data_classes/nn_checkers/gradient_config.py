from dataclasses import dataclass


@dataclass
class GradientConfig:
    period: int = 0
    sample_size: int = 3
    delta: float = 0.0001
    relative_err_max_thresh: float = 0.01
