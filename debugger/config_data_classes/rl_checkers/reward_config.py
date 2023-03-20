from dataclasses import dataclass


@dataclass
class Fluctuation:
    """
    disabled: disable the check or not
    fluctuation_rmse_min: the minimum value of the RMSE to consider the reward values fluctuating
    """

    disabled: bool = False
    fluctuation_rmse_min: float = 0.1


@dataclass
class Monotonicity:
    """
    disabled: disable the check or not
    stagnation_thresh: the minimum value of the RMSE to consider the reward values fluctuating
    reward_stagnation_tolerance: the percentage of reward tolerance accepted to consider the agent close to reach its goal
    """

    disabled: bool = False
    stagnation_thresh: float = 0.25
    reward_stagnation_tolerance: float = 0.01


@dataclass
class RewardConfig:
    """
    period: represents the number of calls required to run the checks
    skip_run_threshold: number of steps to skip during the training
    exploration_perc: The percentage of steps where the agent is considered exploring
    exploitation_perc:  The percentage of steps where the agent is considered exploring
    start: the minimum number of elements required to be in the buffer
    window_size: the number of points to be aggregated
    """

    period: int = 100
    exploration_perc: float = 0.2
    exploitation_perc: float = 0.8
    start: int = 5
    window_size: int = 3
    fluctuation: Fluctuation = Fluctuation()
    monotonicity: Monotonicity = Monotonicity()
