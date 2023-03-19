from dataclasses import dataclass


@dataclass
class Normalization:
    """
    disabled: disable the check or not
    normalized_reward_min: The normalization min boundary
    normalized_reward_max: The normalization max boundary
    """
    disabled: bool = False
    normalized_reward_min: float = -10.0
    normalized_reward_max: float = 10.0


@dataclass
class EnvironmentConfig:
    """
    period: represents the number of calls required to run the checks (0 means that it will only run once before starting the training)
    observations_std_coef_thresh: The min std value to evaluate the variability of the observations
    normalization
    """
    period: int = 0
    observations_std_coef_thresh: float = 0.001
    normalization: Normalization = Normalization()
