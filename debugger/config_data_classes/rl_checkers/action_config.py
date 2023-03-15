from dataclasses import dataclass


@dataclass
class LowStart:
    disabled: bool = False
    start: int = 3
    entropy_min_thresh: float = 0.3


@dataclass
class Monotonicity:
    disabled: bool = False
    increase_thresh: float = 0.1
    stagnation_thresh: float = 1e-3


@dataclass
class StrongDecrease:
    disabled: bool = False
    strong_decrease_thresh: int = 5
    acceleration_points_ratio: float = 0.2


@dataclass
class Fluctuation:
    disabled: bool = False
    fluctuation_thresh: float = 0.5


@dataclass
class ActionStag:
    disabled: bool = False
    start: int = 100
    similarity_pct_thresh: float = 0.8


@dataclass
class ActionStagPerEp:
    disabled: bool = False
    nb_ep_to_check: int = 2
    last_step_num: int = 50
    reward_tolerance: float = 0.5


@dataclass
class ActionConfig:
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
