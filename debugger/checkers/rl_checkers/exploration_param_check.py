import numpy as np
import torch
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import get_data_slope


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "period": 1000,
        "starting_value": 1,
        "ending_value": 0,
        "check_initialization": {"disabled": False},
        "check_monotonicity": {"disabled": False},
        "check_quick_change": {"disabled": False, "strong_decrease_thresh": 5, "acceleration_points_ratio": 0.5}
    }

    return config


class ExplorationParameterCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="ExplorationParameter", config=get_config())
        self.exploration_factor_buffer = []

    def run(self, exploration_factor) -> None:
        """
        Checks the evolution of the exploration parameter during training. A good exploration strategy is one that
        starts with a high exploration rate to encourage the agent to try different actions and then gradually
        reduces the exploration rate to shift towards exploitation. This function performs the following checks:

        (1) Ensures that the initial value of the exploration parameter is set correctly.
        (2) Verifies that the exploration parameter is decreasing over time as the agent learns.
        (3) Checks if the exploration parameter is changing too rapidly, which can lead to unstable behavior.

        Args:
            exploration_factor (float): the value of the exploration parameter
        """
        if self.is_final_step():
            self.exploration_factor_buffer += [exploration_factor]
        self.check_initial_value()
        self.check_exploration_parameter_monotonicity()
        self.check_is_changing_too_quickly()

    def check_initial_value(self):
        """
        Checks if the initial value is correctly set.
        """
        if (len(self.exploration_factor_buffer) == 1) and not self.config["check_initialization"]["disabled"]:
            if self.exploration_factor_buffer[0] != self.config["starting_value"]:
                self.error_msg.append(self.main_msgs['bad_exploration_param_initialization'].format(
                    self.exploration_factor_buffer[0], self.config["starting_value"]))

    def check_exploration_parameter_monotonicity(self):
        """
        Checks whether the exploration parameter value is grdually decreasing or increasing depending on its starting
        and ending values
        """
        if self.config["check_quick_change"]["disabled"]:
            return
        if self.check_period():
            slope = get_data_slope(torch.tensor(self.exploration_factor_buffer, device='cuda'))[0]
            if (slope > 0) and (self.config["starting_value"] > self.config["ending_value"]):
                self.error_msg.append(self.main_msgs['increasing_exploration_factor'])
            elif (slope < 0) and (self.config["starting_value"] < self.config["ending_value"]):
                self.error_msg.append(self.main_msgs['decreasing_exploration_factor'])

    def check_is_changing_too_quickly(self):
        """
        Checks if the exploration parameter's value is changing too quickly
        """
        if self.config["check_quick_change"]["disabled"]:
            return
        if self.check_period():
            abs_second_derivative = np.abs(np.gradient(np.gradient(self.exploration_factor_buffer)))
            if np.any(abs_second_derivative > self.config["check_quick_change"]["strong_decrease_thresh"]):
                    self.error_msg.append(self.main_msgs['quick_changing_exploration_factor'])
