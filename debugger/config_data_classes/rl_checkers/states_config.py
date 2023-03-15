from dataclasses import dataclass


@dataclass
class ResetConfig:
    disabled: bool = False


@dataclass
class NormalizationConfig:
    disabled: bool = False
    normalized_data_min: float = -10
    normalized_data_max: float = 10


@dataclass
class StagnationConfig:
    disabled: bool = False
    period: int = 500


@dataclass
class StatesConvergenceConfig:
    disabled: bool = False
    start: int = 10
    last_obs_num: int = 10
    reward_tolerance: float = 0.5
    final_eps_perc: float = 0.2


@dataclass
class StatesConfig:
    period: int = 1
    skip_run_threshold: int = 2
    exploitation_perc: float = 0.8
    reset: ResetConfig = ResetConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    stagnation: StagnationConfig = StagnationConfig()
    states_convergence: StatesConvergenceConfig = StatesConvergenceConfig()
