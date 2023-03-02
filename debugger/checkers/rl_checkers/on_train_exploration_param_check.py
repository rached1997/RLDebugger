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
        "starting_value": 1,
        "ending_value": 0,
        "check_initialization": {"disabled": False},
        "check_monotonicity": {"disabled": False},
        "check_quick_change": {"disabled": False, "strong_decrease_thresh": 0.05}
    }

    return config


class OnTrainExplorationParameterCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainExplorationParameter", config=get_config())
        self.exploration_factor_buffer = []

    def run(self, exploration_factor) -> None:
        # todo DEBUG: debug this and check if it's to slow, also this may not work because the epsilon in training and testing are not the same
        if self.is_final_step_of_ep():
            self.exploration_factor_buffer += [exploration_factor]
        self.check_initial_value()
        self.check_exploration_parameter_monotonicity()
        self.check_is_changing_too_quickly()

    def check_initial_value(self):
        if (self.iter_num == 1) and not self.config["check_initialization"]["disabled"]:
            if self.exploration_factor_buffer[0] != self.config["starting_value"]:
                self.error_msg.append(self.main_msgs['bad_exploration_param_initialization'].format(
                    self.exploration_factor_buffer[0], self.config["starting_value"]))

    def check_exploration_parameter_monotonicity(self):
        if self.config["check_quick_change"]["disabled"]:
            return
        if self.check_period():
            derivative = np.diff(self.exploration_factor_buffer)
            if all(x > 0 for x in derivative) and (self.config["starting_value"] > self.config["ending_value"]):
                self.error_msg.append(self.main_msgs['increasing_exploration_factor'])
            elif all(x < 0 for x in derivative) and (self.config["starting_value"] < self.config["ending_value"]):
                self.error_msg.append(self.main_msgs['decreasing_exploration_factor'])
            else:
                self.error_msg.append(self.main_msgs['stagnating_exploration_factor'])

    def check_is_changing_too_quickly(self):
        if self.config["check_quick_change"]["disabled"]:
            return
        if self.check_period():
            second_derivative = np.diff(np.diff(self.exploration_factor_buffer))
            if all(x > self.config["check_quick_change"]["strong_decrease_thresh"] for x in second_derivative) and (
                    self.config["starting_value"] > self.config["ending_value"]):
                if self.config["starting_value"] > self.config["ending_value"]:
                    self.error_msg.append(self.main_msgs['fast_decreasing_exploration_factor'])
                else:
                    self.error_msg.append(self.main_msgs['fast_increasing_exploration_factor'])
