import torch

from debugger.config_data_classes.rl_checkers.value_function_config import ValueFunctionConfig
from debugger.debugger_interface import DebuggerInterface


class ValueFunctionCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="ValueFunction", config=ValueFunctionConfig)

    def run(
        self, targets, steps_rewards, discount_rate, predicted_next_vals, steps_done
    ) -> None:
        """
        -----------------------------------   I. Introduction of the Value function Check   -----------------------------------

        This check is particularly useful for DRL applications that use Q-value based learning, as the Q-value is
        used to calculate the Bellman equation. Accurately calculating the Bellman equation is crucial to avoid
        errors and ensure optimal learning efficiency of the DRL agent.
        Therefore, this check is responsible for verifying that the Bellman equation is being applied correctly

        ------------------------------------------   II. The performed checks  -----------------------------------------

        The value function check performs the following check:
            (1) Checks whether the bellman equation is applied correctly.

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are:
            - A coding error : The bellman is not calculated correctly (checks triggered : 1)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues :
            - Fix the calculation of the Bellman function (checks that can be fixed: 1)

        Args:
            targets: the actual next q values
            steps_rewards: the rewards received in each step of the data used to predicted predicted_next_vals
            discount_rate: the gamma value required to apply the bellman equation
            predicted_next_vals: the predicted next q values
            steps_done: the list of flags of each step being done or not

        """
        q_targets = steps_rewards + discount_rate * predicted_next_vals * (
            1 - steps_done
        )
        if not torch.equal(targets, q_targets):
            self.error_msg.append(self.main_msgs["val_func_err"])
