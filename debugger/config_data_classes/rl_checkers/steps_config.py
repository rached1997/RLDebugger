from dataclasses import dataclass

@dataclass
class CheckStagnationConfig:
    disabled: bool = False

@dataclass
class PoorMaxStepPerEpConfig:
    disabled: bool = False
    max_reward_tol: float = 0.1

@dataclass
class StepsConfig:
    period: int = 1
    exploitation_perc: float = 0.8
    check_stagnation: CheckStagnationConfig = CheckStagnationConfig()
    poor_max_step_per_ep: PoorMaxStepPerEpConfig = PoorMaxStepPerEpConfig()
