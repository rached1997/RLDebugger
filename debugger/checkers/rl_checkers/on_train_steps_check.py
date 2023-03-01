import numpy as np
import torch
from debugger.debugger_interface import DebuggerInterface


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 1000,
        "check_stagnation": {"disabled": False},
        "poor_max_step_per_ep": {"disabled": False, "exploitation_perc": 0.8, },
    }

    return config


class OnTrainStepCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainStep", config=get_config())
        self.steps_number_buffer = []
        self.reward_number_buffer = []

    def run(self, step, reward) -> None:
        # We should append this when the episode is done
        self.steps_number_buffer += [step]
        self.reward_number_buffer += [reward]
        self.check_step_is_not_changing()

        # Todo add the check that the max steps is too low



    def check_step_is_not_changing(self):
        if self.config["check_stagnation"]["disabled"]:
            return
        if self.steps_number_buffer.count(self.steps_number_buffer[0]) == len(self.steps_number_buffer):
            self.error_msg.append(self.main_msgs['steps_are_not_changing'])
