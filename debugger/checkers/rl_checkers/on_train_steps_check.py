import statistics

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
        "Period": 10,
        "check_stagnation": {"disabled": False},
        "poor_max_step_per_ep": {"disabled": False, "final_eps_perc": 0.2, "max_reward_tol": 0.1, },
    }

    return config


class OnTrainStepCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainStep", config=get_config())
        self.final_step_number_buffer = []
        self.reward_number_buffer = []

    def run(self, reward, max_reward, max_total_steps) -> None:
        if self.is_final_step_of_ep():
            self.final_step_number_buffer += [self.step_num]
            self.reward_number_buffer += [reward]
        self.check_step_is_not_changing( max_reward, max_total_steps)

    def check_step_is_not_changing(self, max_reward, max_total_steps):
        if self.config["check_stagnation"]["disabled"]:
            return

        if self.check_period() and (
                self.step_num >= (max_total_steps * (1 - self.config["poor_max_step_per_ep"]["final_eps_perc"]))):
            if (statistics.mean(self.final_step_number_buffer) >= self.max_steps_per_episode) and \
                    (statistics.mean(self.final_step_number_buffer) <
                     (max_reward * self.config["poor_max_step_per_ep"]["max_reward_tol"])):
                self.error_msg.append(self.main_msgs['poor_max_ste_per_ep'])
