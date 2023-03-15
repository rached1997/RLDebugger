from dataclasses import dataclass


@dataclass
class Fluctuation:
    disabled: bool = False
    fluctuation_rmse_min: float = 0.1


@dataclass
class Monotonicity:
    disabled: bool = False
    stagnation_thresh: float = 0.25
    reward_stagnation_tolerance: float = 0.01
    stagnation_episodes: int = 20


@dataclass
class RewardConfig:
    period: int = 100
    skip_run_threshold: int = 2
    exploration_perc: float = 0.2
    exploitation_perc: float = 0.8
    start: int = 5
    window_size: int = 3
    fluctuation: Fluctuation = Fluctuation()
    monotonicity: Monotonicity = Monotonicity()

