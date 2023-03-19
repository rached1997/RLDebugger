from dataclasses import dataclass


@dataclass
class ResetConfig:
    disabled: bool = False


@dataclass
class NormalizationConfig:
    """
    disabled: disable the check or not
    normalized_data_min: the min value of the observations
    normalized_data_max: the max value of the observations
    """
    disabled: bool = False
    normalized_data_min: float = -10
    normalized_data_max: float = 10


@dataclass
class StagnationConfig:
    """
    disabled: disable the check or not
    stagnated_obs_nbr: the number of observations to compare
    """
    disabled: bool = False
    stagnated_obs_nbr: int = 10


@dataclass
class StatesConvergenceConfig:
    """
    disabled: disable the check or not
    number_of_episodes: number of episodes to compare
    last_obs_num: the number of observations to compare
    reward_tolerance: the percentage of reward tolerance to consider the agent close to achieve its goal
    """
    disabled: bool = False
    number_of_episodes: int = 2
    last_obs_num: int = 10
    reward_tolerance: float = 0.5


@dataclass
class StatesConfig:
    """
    period: represents the number of calls required to run the checks
    skip_run_threshold: number of steps to skip during the training
    exploitation_perc:  The percentage of steps where the agent is considered exploring
    start: the minimum number of elements required to be in the buffer
    window_size: the number of points to be aggregated
    """
    period: int = 1
    skip_run_threshold: int = 2
    exploitation_perc: float = 0.8
    start = 3
    reset: ResetConfig = ResetConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    stagnation: StagnationConfig = StagnationConfig()
    states_convergence: StatesConvergenceConfig = StatesConvergenceConfig()
