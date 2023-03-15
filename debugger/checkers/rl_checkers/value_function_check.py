import torch
from debugger.debugger_interface import DebuggerInterface


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "period": 1,
    }
    return config


class ValueFunctionCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="ValueFunction", config=get_config())

    def run(
        self, targets, steps_rewards, discount_rate, predicted_next_vals, steps_done
    ) -> None:
        """
        This checks is usefull for the DRL applicactions that uses q-value based learning. The q-value is calculated
        based on the bellman equation. Which is generally implemented manually by the developper. The goal of this
        check is to make sure that the bellman equation is being applied correctly.

        The value function check performs the following check:
            (1) Checks whether the bellman equation is applied correctly.

        The potential root causes behind the warnings that can be detected are:
            - A coding error : The bellman is not calculated correctly (checks triggered : 1)

        The recommended fixes for the detected issues :
            - Fix the calculation of the Bellman function (checks that can be fixed: 1)

        Args:
            targets: the actual next q values
            steps_rewards: the rewards received in each step of the data used to predicted predicted_next_vals
            discount_rate: the gamma value required to apply the bellman equation
            predicted_next_vals: the predicted next q values
            steps_done: the list of flags of each step being done or not

        Returns:

        """
        q_targets = steps_rewards + discount_rate * predicted_next_vals * (
            1 - steps_done
        )
        if not torch.equal(targets, q_targets):
            self.error_msg.append(self.main_msgs["val_func_err"])
