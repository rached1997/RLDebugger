import torch
from debugger.debugger_interface import DebuggerInterface
import numpy as np

from debugger.utils.utils import smoothness


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 1, }
    return config


class OnTrainValueFunctionCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainValueFunction", config=get_config())

    def run(self, targets, steps_rewards, discount_rate, predicted_next_vals, steps_done) -> None:
        q_targets = steps_rewards + discount_rate * predicted_next_vals * (1 - steps_done)
        if not torch.equal(targets, q_targets):
            self.error_msg.append(self.main_msgs['val_func_err'])
