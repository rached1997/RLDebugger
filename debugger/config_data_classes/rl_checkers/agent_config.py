from dataclasses import dataclass


@dataclass
class TargetUpdateConfig:
    """
    disabled: disable the check or not
    """

    disabled: bool = False


@dataclass
class SimilarityConfig:
    """
    disabled: disable the check or not
    """

    disabled: bool = False


@dataclass
class WrongModelOutConfig:
    """
    disabled: disable the check or not
    """

    disabled: bool = False


@dataclass
class KLdivConfig:
    """
    disabled: disable the check or not
    div_threshold: the threshold of the KL divergence
    """

    disabled: bool = False
    div_threshold: float = 0.1


@dataclass
class AgentConfig:
    """
    period: represents the number of calls required to run the checks
    start: the minimum number of elements required to be in the buffer
    target_update
    similarity
    wrong_model_out
    kl_div
    """

    period: int = 100
    start: int = 32
    target_update: TargetUpdateConfig = TargetUpdateConfig()
    similarity: SimilarityConfig = SimilarityConfig()
    wrong_model_out: WrongModelOutConfig = WrongModelOutConfig()
    kl_div: KLdivConfig = KLdivConfig()
