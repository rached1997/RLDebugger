import numpy as np
import torch
from debugger.config_data_classes.nn_checkers.loss_config import LossConfig
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
from debugger.utils.utils import smoothness


class LossCheck(DebuggerInterface):
    """
    The check is in charge of verifying the loss function during training.
    """

    def __init__(self):
        super().__init__(check_type="Loss", config=LossConfig)
        self.min_loss = np.inf
        self.current_losses = []
        self.average_losses = []

    def run(
        self,
        targets: torch.Tensor,
        training_observations: torch.Tensor,
        loss_fn: torch.nn.Module,
        model: torch.nn.Module,
        actions: torch.Tensor,
    ) -> None:
        """
        The loss in Deep Learning in general is an indicator of how accurate is the neural network, and generally the
        expected ehaviour of the loss function is that it starts from a high value and keeps on decreasing until
        reaching 0. Howeve, in Deep Reinforcement Learning, the loss doesn't have the same impact as in the deep
        learning, since the goal of the DRL agent is to try to reach an expected maximum reward. But, the loss still
        plays a key role in updating the agents in DRL that consists of one or multiple neural networks in
        interaction.The loss in this context is the value that indicates to the neural networks whether they are
        close to reach their expected goals or not. An unstable behaviour in the loss function can indicate many
        potential issues that are detabilizing the learning or reducing the performance of the agnet. For example,
        in many cases in DRL, it's recommandedto use two seperate copies of the netwprk, called main and target
        network, that are updated in different periods. This helps the agent stabilize more its learning. One error
        that can occureis choosing a low update period, this can be detected by the behaviour of the loss which would
        be fluctuating a lot

        This loss check class performs multiple checks on the loss function during the training:

        (1) run the following pre-checks:
            a. Ensure correct reduction of the loss function. this is more useful for custom loss
            function implementations. Loss is calculated with increasing batch sizes to confirm proportional increase,
            indicating that the loss estimation is based on average, not sum.

            b. Verify that the optimizer's initial loss is as expected. An initial loss approximation can be
            calculated for most of the predefined loss functions. This function compares the model's initial loss to the
            calculated approximation.

        (2) Check the numerical instabilities of loss values. (check whehter they havenan or infinite values)
        (3) Check the abnormal loss curvature of the loss (check the function check_loss_curve for more details)
            a. Non-or Slow-Decreasing loss.
            b. Diverging loss
            c. Highly-Fluctuating loss

        The potential root causes behind the warnings that can be detected are
            - Using a bad loss function (checks triggered : 1,2,3)
            - Wrong implementation of the loss function (checks triggered : 1,2,3)
            - An unstable learning process (checks triggered : 2,3) :
                * Bad hyperparameters values
                * Low update period of the target network (a probel related to the agent in DRL)
                * Bad architecture of the model

        The recommended fixes for the detected issues:
            - Change the loss function ( checks tha can be fixed: 1,2,3)
            - Verify that the loss function works correctly (checks tha can be fixed: 1,2,3)
            - Increase the update period of the target network (checks tha can be fixed: 2,3)
            - Change the architecture of the neural network (checks tha can be fixed: 2,3)

        Args:
        targets (Tensor): A sample of targets collected periodically during the training.
        predictions (Tensor): A sample of predictions collected periodically during the training.
        loss_fn (torch.nn.Module): the loss function of the model.
        """
        actions_probs = model(training_observations)
        predictions = actions_probs[torch.arange(actions_probs.size(0)), actions]
        if self.iter_num == 1:
            self.run_pre_checks(targets, predictions, loss_fn, model)
        loss_val = float(get_loss(predictions, targets, loss_fn))
        if self.check_numerical_instabilities(loss_val):
            return
        self.current_losses += [loss_val]
        if self.check_period():
            losses = self.update_losses(np.mean(self.current_losses))
            self.check_loss_curve(losses)
            self.current_losses = []

    def update_losses(self, curr_loss: np.ndarray) -> np.ndarray:
        """
        Updates the array of average loss values with new averaged loss value (over a window size).

        Args:
            curr_loss (np.ndarray): the average loss values over a window size.

        Returns
            (numpy.ndarray) : The array of all average (smoothed) loss values.

        """
        self.min_loss = min(curr_loss, self.min_loss)
        self.average_losses += [curr_loss]
        return np.array(self.average_losses)

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
                losses[n_losses - i] / self.min_loss
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
