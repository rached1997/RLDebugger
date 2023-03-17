import numpy as np
import torch
from debugger.config_data_classes.nn_checkers.loss_config import LossConfig
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
from debugger.utils.utils import smoothness


class LossCheck(DebuggerInterface):
    """
    The check is in charge of verifying the loss function during training.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
            * min_loss: the minimum loss value encountered
            * current_losses : A list collecting the current loss
            * average_losses : A list collecting the current loss
        """
        super().__init__(check_type="Loss", config=LossConfig)
        self._min_loss = np.inf
        self._current_losses = []
        self._average_losses = []

    def run(
        self,
        targets: torch.Tensor,
        training_observations: torch.Tensor,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
        actions: torch.Tensor,
    ) -> None:
        """
        -----------------------------------   I. Introduction of the loss Check  -----------------------------------

        This class performs checks on the loss function in Deep Reinforcement Learning (DRL). In DRL,
        the loss function plays a key role in updating the agent's neural networks and guiding them towards the
        expected maximum reward. While the loss function in DRL is different from the loss in Deep Learning,
        it still needs to exhibit stable behavior to ensure efficient learning.

        The expected behavior of the loss function is that it starts from a high value and gradually decreases until
        it reaches 0. However, an unstable loss function can indicate potential issues that are destabilizing the
        learning or reducing the agent's performance.

        For example,in many DRL applications, it is recommended to use two separate copies of the network,
        known as the main and target network, which are updated at different periods. This technique helps stabilize
        the agent's learning process. However, choosing a low update period for the target network can lead to an
        unstable learning, which can be detected by observing the behavior of the loss function. In such cases,
        the loss would fluctuate excessively, indicating that the agent's performance is being negatively impacted by
        the update frequency of the target network.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        The loss check class performs the following checks on the loss function during the training:
        (1) run the following pre-checks:

            a. Ensure correct reduction of the loss function (i.e. the loss estimation is based on average,
            not sum). This is more useful for custom loss function implementations.
            b. Verify that the optimizer's initial loss is as expected. An initial loss approximation can be
            calculated for most of the predefined loss functions. This function compares the model's initial loss to the
            calculated approximation.
        (2) Check the numerical instabilities of loss values. (check whether they have nan or infinite values)
        (3) Check the abnormal loss curvature of the loss (check the function check_loss_curve for more details)
            a. Non-or Slow-Decreasing loss.
            b. Diverging loss
            c. Highly-Fluctuating loss

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are
            - Using a bad loss function (checks triggered : 1,2,3)
            - Wrong implementation of the loss function (checks triggered : 1,2,3)
            - An unstable learning process (checks triggered : 2,3) :
                * Bad hyperparameters values
                * Low update period of the target network (a probel related to the agent in DRL)
                * Bad architecture of the model

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues:
            - Change the loss function ( checks tha can be fixed: 1,2,3)
            - Verify that the loss function works correctly (checks tha can be fixed: 1,2,3)
            - Increase the update period of the target network (checks tha can be fixed: 2,3)
            - Change the architecture of the neural network (checks tha can be fixed: 2,3)

        Examples
        --------
        To perform loss checks, the debugger needs to be called when updating the agent.

        >>> from debugger import rl_debugger
        >>> ...
        >>> next_qvals = target_qnet(next_states)
        >>> next_qvals, _ = torch.max(next_qvals, dim=1)
        >>> batch = replay_buffer.sample(batch_size=32)
        >>> q_targets = batch["reward"] + discount_rate * next_qvals * (1 - batch["done"])
        >>> rl_debugger.debug(training_observations=batch["state"], targets=q_targets.detach(), actions=actions,
        >>>                   model=qnet, loss_fn=loss_fn)
        >>> loss = loss_fn(pred_qvals, q_targets).mean()

        Args:
        targets (Tensor): A sample of targets collected periodically during the training.
        training_observations(Tensor): the batch of observations collected during the training used to obtain
            actions_probs
        loss_fn (torch.nn.Module): the loss function of the model.
        model (nn.Module): the main model.
        actions (Tensor): A sample of actions collected periodically during the training.
        """
        actions_probs = model(training_observations)
        predictions = actions_probs[torch.arange(actions_probs.size(0)), actions]
        if self.iter_num == 1:
            self.run_pre_checks(targets, predictions, loss_fn, model)
        loss_val = float(get_loss(predictions, targets, loss_fn))
        if self.check_numerical_instabilities(loss_val):
            return
        self._current_losses += [loss_val]
        if self.check_period():
            losses = self.update_losses(np.mean(self._current_losses))
            self.check_loss_curve(losses)
            self._current_losses = []

    def update_losses(self, curr_loss: np.ndarray) -> np.ndarray:
        """
        Updates the array of average loss values with new averaged loss value (over a window size).

        Args:
            curr_loss (np.ndarray): the average loss values over a window size.

        Returns
            (numpy.ndarray) : The array of all average (smoothed) loss values.

        """
        self._min_loss = min(curr_loss, self._min_loss)
        self._average_losses += [curr_loss]
        return np.array(self._average_losses)

    def check_numerical_instabilities(self, loss_value: float) -> bool:
        """
        Validates the numerical stability of loss value during training.

        Args:
            loss_value (float): the current loss value.

        Returns:
            (bool): True if there is any NaN or infinite value present, False otherwise.
        """
        if self.config.numeric_ins.disabled:
            return False
        if np.isnan(loss_value):
            self.error_msg.append(self.main_msgs["nan_loss"])
            return True
        if np.isinf(loss_value):
            self.error_msg.append(self.main_msgs["inf_loss"])
            return True
        return False

    def check_loss_curve(self, losses: np.ndarray) -> None:
        """
        Check the abnormal loss curvature of the loss. The shape and dynamics of a loss curve can help diagnose
        the behavior of the optimizer against the learning problem (more details can be found
        here: https://cs231n.github.io/neural-networks-3/. This check verify the following abnormalities:
            - Non- or Slow-Decreasing loss.
            - Diverging loss
            - Highly-Fluctuating loss

        Args:
            losses: (numpy.ndarray) : average (smoothed) loss values.

        Returns:
            None

        """
        n_losses = len(losses)
        if n_losses >= self.config.non_dec.window_size:
            dec_pers = np.array(
                [
                    (losses[-i - 1] - losses[-i]) / losses[-i - 1]
                    for i in range(1, self.config.non_dec.window_size)
                ]
            )
            if (dec_pers < self.config.non_dec.decr_percentage).all() and not (
                self.config.non_dec.disabled
            ):
                self.error_msg.append(self.main_msgs["stagnated_loss"])
        if n_losses >= self.config.div.window_size:
            abs_loss_incrs = [
                losses[n_losses - i] / self._min_loss
                for i in range(self.config.div.window_size, 0, -1)
            ]
            inc_rates = np.array(
                [
                    abs_loss_incrs[-i] / abs_loss_incrs[-i - 1]
                    for i in range(1, self.config.div.window_size)
                ]
            )
            if (inc_rates >= self.config.div.incr_abs_rate_max_thresh).all() and not (
                self.config.div.disabled
            ):
                self.error_msg.append(self.main_msgs["div_loss"].format(max(inc_rates)))
        # if n_losses >= self.config['fluct']['window_size']:
        smoothness_val = smoothness(losses[-self.config.fluct.window_size :])
        if smoothness_val < self.config.fluct.smoothness_ratio_min_thresh and not (
            self.config.fluct.disabled
        ):
            self.error_msg.append(
                self.main_msgs["fluctuated_loss"].format(
                    smoothness_val, self.config.fluct.smoothness_ratio_min_thresh
                )
            )

    def run_pre_checks(self, targets, predictions, loss_fn, model):
        """
        Run multiple checks on the loss function and its generated values. This checker runs before the training and
        does the following checks on the initial loss outputs :

        The checks include: 1.Ensure correct reduction of the loss function. this is more useful for custom loss
        function implementations. Loss is calculated with increasing batch sizes to confirm proportional increase,
        indicating that the loss estimation is based on average, not sum.

        2. Verify that the optimizer's initial loss is as expected. An initial loss approximation can be
        calculated for most of the predefined loss functions. This function compares the model's initial loss to the
        calculated approximation.

        Args:
            model: (nn.model) The model being trained
            loss_fn: (torch.nn.Module) the loss function of the model.
            targets (Tensor): The ground truth of the initial observationsTargets used in the loss function (for
             example the targets in the DQN are the Q_target).
            predictions (Tensor): The outputs of the model in the initial set of observations.
            loss_fn (function): The loss function.
            model (nn.Module): The model to be trained.
        """
        losses = []
        n = self.config.init_loss.size_growth_rate
        while n <= (
            self.config.init_loss.size_growth_rate
            * self.config.init_loss.size_growth_iters
        ):
            derived_batch_y = torch.cat([targets] * n, dim=0)
            derived_predictions = torch.cat(n * [predictions], dim=0)
            loss_value = float(get_loss(derived_predictions, derived_batch_y, loss_fn))
            losses.append(loss_value)
            n *= self.config.init_loss.size_growth_rate
        rounded_loss_rates = [
            round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)
        ]
        equality_checks = sum(
            [
                (loss_rate == self.config.init_loss.size_growth_rate)
                for loss_rate in rounded_loss_rates
            ]
        )
        if equality_checks == len(rounded_loss_rates):
            self.error_msg.append(self.main_msgs["poor_reduction_loss"])

        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            initial_loss = float(get_loss(predictions, targets, loss_fn))
            initial_weights, _ = get_model_weights_and_biases(model)
            number_of_actions = list(initial_weights.items())[-1][1].shape[0]
            expected_loss = -torch.log(torch.tensor(1 / number_of_actions))
            err = torch.abs(initial_loss - expected_loss)
            if err >= self.config.init_loss.dev_ratio * expected_loss:
                self.error_msg.append(
                    self.main_msgs["poor_init_loss"].format(
                        round((err / expected_loss), 3)
                    )
                )
