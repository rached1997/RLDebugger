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
        "Period": 1,
        "exploitation_perc": 0.8,
        "check_stagnation": {"disabled": False},
        "poor_max_step_per_ep": {"disabled": False, "max_reward_tol": 0.1},
    }

    return config


class StepCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="Step", config=get_config())
        self.final_step_number_buffer = []
        self.episode_reward_buffer = []
        self.last_step_num = 0

    def run(self, reward, max_reward, max_total_steps, max_steps_per_episode) -> None:
        if self.is_final_step():
            self.final_step_number_buffer += [self.step_num-self.last_step_num]
            self.episode_reward_buffer += [reward]
            self.last_step_num = self.step_num

        self.check_step_is_not_changing(max_reward, max_total_steps, max_steps_per_episode)

    def check_step_is_not_changing(self, max_reward, max_total_steps, max_steps_per_episode):
        if self.config["check_stagnation"]["disabled"]:
            return

        if self.check_period() and (self.step_num >= (max_total_steps * self.config["exploitation_perc"])):
            if (statistics.mean(self.final_step_number_buffer) >= max_steps_per_episode) and \
                    (statistics.mean(self.episode_reward_buffer) <
                     (max_reward * self.config["poor_max_step_per_ep"]["max_reward_tol"])):
                self.error_msg.append(self.main_msgs['poor_max_step_per_ep'])
