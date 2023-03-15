import numpy as np
import torch

from debugger.config_data_classes.rl_checkers.exploration_param_config import ExplorationPramConfig
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import get_data_slope


class ExplorationParameterCheck(DebuggerInterface):
    def __init__(self):
        """
        Initializes the following parameters:
        exploration_factor_buffer : A buffer storing the values of the parameter controlling the ratio of exploration
        and exploitation
        """
        super().__init__(check_type="ExplorationParameter", config=ExplorationPramConfig)
        self.exploration_factor_buffer = []

    def run(self, exploration_factor) -> None:
        """
        Checks the evolution of the exploration parameter during training. A good exploration strategy is one that
        starts with a high exploration rate to encourage the agent to try different actions and then gradually
        reduces the exploration rate to shift towards exploitation. In many exploration strategies there is a
        parameter that cpntrols the ratio of exploration and exploitation. For example in the epsilon greedy the
        epsilon is the one responsible for the exploration exploitation trade off. It's essential to make sure that
        the parameter controlling the exploration is being updated correctly otherwise it would lead to a weak
        exploration or exagerated exploration which will destabilise the behaviour of the learning

        The exploration parameters check performs the following checks:
        (1) Ensures that the initial value of the exploration parameter is set correctly.
        (2) Verifies that the exploration parameter is decreasing (or increasing depending on its start and end values) over time as the agent learns.
        (3) Checks if the exploration parameter is changing too rapidly, which can lead to unstable behavior.

        The potential root causes behind the warnings that can be detected are:
            - Wrong initialization of the exploration parameter (checks triggered : 1)
            - Wrong update rule of the exploration parameter (checks triggered : 1,2,3)
            - Wrong update frequency of the exploration parameter (checks triggered : 3)

        The recommended fixes for the detected issues :
            - Check if the exploration parameter is correctly initialized (checks that can be fixed: 1,2,3)
            - Check if the exploration parameter is being update (checks that can be fixed: 2)
            - Check if the exploration parameter is being update correctly (checks that can be fixed: 2)
            - Change the update frequency of the exploration (checks that can be fixed: 3)
            - Change the update rule of the exploration (checks that can be fixed: 3)

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
        if (len(self.exploration_factor_buffer) == 1) and not self.config[
            "check_initialization"
        ]["disabled"]:
            if self.exploration_factor_buffer[0] != self.config.starting_value:
                self.error_msg.append(
                    self.main_msgs["bad_exploration_param_initialization"].format(
                        self.exploration_factor_buffer[0], self.config.starting_value
                    )
                )

    def check_exploration_parameter_monotonicity(self):
        """
        Checks whether the exploration parameter value is grdually decreasing or increasing depending on its starting
        and ending values
        """
        if self.config.check_quick_change.disabled:
            return
        if self.check_period():
            slope = get_data_slope(
                torch.tensor(self.exploration_factor_buffer, device=self.device)
            )[0]
            if (slope > 0) and (self.config.starting_value > self.config.ending_value):
                self.error_msg.append(self.main_msgs["increasing_exploration_factor"])
            elif (slope < 0) and (
                self.config.starting_value < self.config.ending_value
            ):
                self.error_msg.append(self.main_msgs["decreasing_exploration_factor"])

    def check_is_changing_too_quickly(self):
        """
        Checks if the exploration parameter's value is changing too quickly
        """
        if self.config.check_quick_change.disabled:
            return
        if self.check_period():
            abs_second_derivative = np.abs(
                np.gradient(np.gradient(self.exploration_factor_buffer))
            )
            if np.any(
                abs_second_derivative
                > self.config.check_quick_change.strong_decrease_thresh
            ):
                self.error_msg.append(
                    self.main_msgs["quick_changing_exploration_factor"]
                )
