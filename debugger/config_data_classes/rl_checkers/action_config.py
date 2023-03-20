from dataclasses import dataclass


@dataclass
class LowStart:
    """
    disabled: disable the check or not
    start: the minimum number of elements required to run the check
    entropy_min_thresh: minimum accepted initial entropy value
    """

    disabled: bool = False
    start: int = 3
    entropy_min_thresh: float = 0.3


@dataclass
class Monotonicity:
    """
    disabled: disable the check or not
    increase_thresh: maximum accepted entropy slope
    stagnation_thresh: minimum accepted entropy slope
    """

    disabled: bool = False
    increase_thresh: float = 0.1
    stagnation_thresh: float = 1e-3


@dataclass
class StrongDecrease:
    """
    disabled: disable the check or not
    strong_decrease_thresh: minimum accepted second derivative of entropy
    acceleration_points_ratio: accepted ratio of entropy point with low second derivative values
    region_length: length of the region on which the check is performed
    """

    disabled: bool = False
    strong_decrease_thresh: int = -0.1
    acceleration_points_ratio: float = 0.2
    region_length: int = 10


@dataclass
class Fluctuation:
    """
    disabled: disable the check or not
    fluctuation_thresh: maximum accepted value of rmse value of the entropy
    region_length: length of the region on which the check is performed
    """

    disabled: bool = False
    fluctuation_thresh: float = 0.1
    region_length: int = 10


@dataclass
class ActionStag:
    """
    disabled: disable the check or not
    start: the minimum number of elements required to run the check
    similarity_pct_thresh: maximum accepted ratio of single action in one episode
    """

    disabled: bool = False
    start: int = 100
    similarity_pct_thresh: float = 0.8


@dataclass
class ActionStagPerEp:
    """
    disabled: disable the check or not
    nb_ep_to_check: number of episode to check
    last_step_num: number of last steps to consider in one episode
    reward_tolerance: reward tolerance
    """

    disabled: bool = False
    nb_ep_to_check: int = 2
    last_step_num: int = 10
    reward_tolerance: float = 0.1


@dataclass
class ActionConfig:
    """
    period: represents the number of calls required to run the checks
    start: the minimum number of elements required to run the checks
    skip_run_threshold: the number of runs to skip in order to accelerate the debugger
    exploration_perc: ratio of first episodes to consider as exploration episodes
    exploitation_perc: ratio of episodes after which we consider that exploration is occurring
    """

    period: int = 50
    start: int = 10
    skip_run_threshold: int = 2
    exploration_perc: float = 0.2
    exploitation_perc: float = 0.8
    low_start: LowStart = LowStart()
    monotonicity: Monotonicity = Monotonicity()
    strong_decrease: StrongDecrease = StrongDecrease()
    fluctuation: Fluctuation = Fluctuation()
    action_stag: ActionStag = ActionStag()
    action_stag_per_ep: ActionStagPerEp = ActionStagPerEp()
