from dataclasses import dataclass


class InstanceWiseOperation:
    sample_size: int = 32
    trials: int = 10


@dataclass
class ProperFittingConfig:
    disabled = True
    period: int = 0
    single_batch_size: int = 8
    total_iters: int = 10
    abs_loss_min_thresh: float = 1e-8
    loss_min_thresh: float = 0.00001
    smoothness_max_thresh: float = 0.95
    mislabeled_rate_max_thresh: float = 0.05
    mean_error_max_thresh: float = 0.001
    sample_size_of_losses: int = 10
    Instance_wise_Operation: InstanceWiseOperation = InstanceWiseOperation()
