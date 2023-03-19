import torch

from debugger.config_data_classes.rl_checkers.q_target_config import QTargetConfig
from debugger.debugger_interface import DebuggerInterface


class QTargetCheck(DebuggerInterface):
    """
    This class of checks is dedicated to Q-Learning-based RL algorithms. It checks whether the Q-learning target values
    are computed correctly.
    For more details on the specific checks performed, refer to the `run()` function.
    """
    def __init__(self):
        super().__init__(check_type="ValueFunction", config=QTargetConfig)

    def run(
        self, targets, steps_rewards, discount_rate, predicted_next_vals, steps_done
    ) -> None:
        """
        -----------------------------------   I. Introduction of the Q Targets Check   ----------------------------

        This check is particularly useful for DRL applications that use Q-value based learning. This check is
        responsible for verifying that the Q-learning target values are computed correctly. the Q-learning target
        values q_targets are used to update the Q-network parameters. We check if the q_targets tensor
        is computed correctly, by comparing it to the provided targets tensor. If the two tensors are not equal, it
        means that there is an error in the computation of the Q-learning targets, which would lead to incorrect
        updates of the Q-network parameters.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        The Q target check performs the following check:
            (1) Checks whether Q-learning target values are computed correctly.

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are:
            - A coding error : The formula is not calculated correctly (checks triggered : 1)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues :
            - Fix the calculation of Q-learning target values (checks that can be fixed: 1)

        Examples
        --------
        To perform Q-learning target values checks, the debugger needs to be called when updating the main and target
        networks.
        Note that the debugger needs to be called jsut after computing the Q targets.

        >>> from debugger import rl_debugger
        >>> ...
        >>> next_qvals = target_qnet(next_states)
        >>> next_qvals, _ = torch.max(next_qvals, dim=1)
        >>> batch = replay_buffer.sample(batch_size=32)
        >>> q_targets = batch["reward"] + discount_rate * next_qvals * (1 - batch["done"])
        >>> rl_debugger.debug(targets=q_targets.detach(), steps_rewards=batch["reward"], discount_rate=discount_rate,
        >>>                   predicted_next_vals=next_qvals.detach(), steps_done=batch["done"])
        >>> loss = loss_fn(pred_qvals, q_targets).mean()

        Args:
            targets: the actual next q values
            steps_rewards: the rewards received in each step of the data used to predicted predicted_next_vals
            discount_rate: the gamma value required to apply the bellman equation
            predicted_next_vals: the predicted next q values
            steps_done: the list of flags of each step being done or not

        """
        if not self.check_period():
            return
        q_targets = steps_rewards + discount_rate * predicted_next_vals * (
            1 - steps_done
        )
        if not torch.equal(targets, q_targets):
            self.error_msg.append(self.main_msgs["q_target_err"])
