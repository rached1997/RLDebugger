from dataclasses import dataclass


@dataclass
class Normalization:
    disabled: bool = False
    normalized_reward_min: float = -10.0
    normalized_reward_max: float = 10.0

@dataclass
class EnvironmentConfig:
    period: int = 0
    observations_std_coef_thresh: float = 0.001
    normalization: Normalization = Normalization()
