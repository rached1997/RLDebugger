from dataclasses import dataclass


@dataclass
class PoorMaxStepPerEpConfig:
    """
    disabled: disable the check or not
    max_reward_tol: reward tolerance
    region_length: length of the region on which the check is performed
    """
    disabled: bool = False
    max_reward_tol: float = 0.1
    region_length: int = 10


@dataclass
class StepsConfig:
    """
    period: represents the number of calls required to run the checks
    skip_run_threshold: the number of runs to skip in order to accelerate the debugger
    exploitation_perc: ratio of episodes after which we consider that exploration is occurring
    """
    period: int = 1
    skip_run_threshold: int = 2
    exploitation_perc: float = 0.8
    poor_max_step_per_ep: PoorMaxStepPerEpConfig = PoorMaxStepPerEpConfig()
