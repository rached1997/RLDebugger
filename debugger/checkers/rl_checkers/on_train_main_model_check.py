import torch
from debugger.debugger_interface import DebuggerInterface
import hashlib


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 100,
        "target_update": {"disabled": False},
        "similarity": {"disabled": False}
    }
    return config


class OnTrainMainModelCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainMainModel", config=get_config())

    def run(self, model, predictions, observations, actions) -> None:
        if (not (predictions is None)) and (not (observations is None)):
            pred_qvals = model(observations)
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]
            if not torch.equal(predictions, pred_qvals):
                self.error_msg.append(self.main_msgs['using_the_wrong_network'])



