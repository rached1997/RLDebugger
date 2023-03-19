from dataclasses import dataclass


@dataclass
class UncertaintyActionConfig:
    """
    period: represents the number of calls required to run the checks
    start: the minimum number of elements required to run the checks
    skip_run_threshold: the number of runs to skip in order to accelerate the debugger
    num_repetitions: number of time to run the model with Monte Carlo dropout layers
    std_threshold: minimum value of epistemic uncertainty std
    buffer_max_size: maximum size of observations buffer
    batch_size: batch size of observations
    """
    period: int = 1000
    start: int = 1000
    skip_run_threshold: int = 2
    num_repetitions: int = 100
    std_threshold: float = 0.5
    buffer_max_size: int = 1000
    batch_size: int = 32
