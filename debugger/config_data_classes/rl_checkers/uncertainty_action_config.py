from dataclasses import dataclass


@dataclass
class UncertaintyActionConfig:
    period: int = 1000
    start: int = 1000
    skip_run_threshold: int = 2
    num_repetitions: int = 100
    std_threshold: float = 0.5
    buffer_max_size: int = 1000
    batch_size: int = 32
