import numpy as np
import torch
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
from debugger.utils.utils import smoothness


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
              "period": 10,
              "numeric_ins": {"disabled": False},
              "non_dec": {"disabled": False, "window_size": 5, "decr_percentage": 0.05},
              "div": {"disabled": False, "incr_abs_rate_max_thresh": 2, "window_size": 5},
              "fluct": {"disabled": False, "window_size": 50, "smoothness_ratio_min_thresh": 0.5},
              "init_loss": {"size_growth_rate": 2, "size_growth_iters": 5, "dev_ratio": 1.0}
    }
    return config


class LossCheck(DebuggerInterface):
    """
    The check is in charge of verifying the loss function during training.
    """

    def __init__(self):
        super().__init__(check_type="Loss", config=get_config())
        self.min_loss = np.inf
        self.current_losses = []
        self.average_losses = []

    def run(self, targets: torch.Tensor, actions_probs: torch.Tensor, loss_fn: torch.nn.Module, model: torch.nn.Module,
            actions: torch.Tensor) -> None:
        """
        This function performs multiple checks on the loss function during the training:

        (1) run the pre-checks described in the function run_pre_checks
        (2) Check the numerical instabilities of loss values. (check the function check_numerical_instabilities
        for more details)
        (3) Check the abnormal loss curvature of the loss (check the function check_loss_curve for more details)

        Args:
        targets (Tensor): A sample of targets collected periodically during the training.
        predictions (Tensor): A sample of predictions collected periodically during the training.
        loss_fn (torch.nn.Module): the loss function of the model.
        """
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
        if self.config['numeric_ins']['disabled']:
            return False
        if np.isnan(loss_value):
            self.error_msg.append(self.main_msgs['nan_loss'])
            return True
        if np.isinf(loss_value):
            self.error_msg.append(self.main_msgs['inf_loss'])
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
        if n_losses >= self.config['non_dec']['window_size']:
            dec_pers = np.array(
                [(losses[-i - 1] - losses[-i]) / losses[-i - 1] for i in
                 range(1, self.config['non_dec']['window_size'])])
            if (dec_pers < self.config['non_dec']['decr_percentage']).all() and not (
                    self.config['non_dec']['disabled']):
                self.error_msg.append(self.main_msgs['stagnated_loss'])
        if n_losses >= self.config['div']['window_size']:
            abs_loss_incrs = [losses[n_losses - i] / self.min_loss for i in range(self.config['div']['window_size'],
                                                                                  0, -1)]
            inc_rates = np.array(
                [abs_loss_incrs[-i] / abs_loss_incrs[-i - 1] for i in
                 range(1, self.config['div']['window_size'])])
            if (inc_rates >= self.config['div']['incr_abs_rate_max_thresh']).all() and not (
                    self.config['div']['disabled']):
                self.error_msg.append(self.main_msgs['div_loss'].format(max(inc_rates)))
        # if n_losses >= self.config['fluct']['window_size']:
        smoothness_val = smoothness(losses[-self.config['fluct']['window_size']:])
        if smoothness_val < self.config['fluct']['smoothness_ratio_min_thresh'] and not (
                self.config['fluct']['disabled']):
            self.error_msg.append(self.main_msgs['fluctuated_loss'].format(smoothness_val,
                                                                           self.config['fluct'][
                                                                               'smoothness_ratio_min_thresh']))

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
        n = self.config["init_loss"]["size_growth_rate"]
        while n <= (self.config["init_loss"]["size_growth_rate"] * self.config["init_loss"]["size_growth_iters"]):
            derived_batch_y = torch.cat([targets] * n, dim=0)
            derived_predictions = torch.cat(n * [predictions], dim=0)
            loss_value = float(get_loss(derived_predictions, derived_batch_y, loss_fn))
            losses.append(loss_value)
            n *= self.config["init_loss"]["size_growth_rate"]
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config["init_loss"]["size_growth_rate"]) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            self.error_msg.append(self.main_msgs['poor_reduction_loss'])

        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            initial_loss = float(get_loss(predictions, targets, loss_fn))
            initial_weights, _ = get_model_weights_and_biases(model)
            number_of_actions = list(initial_weights.items())[-1][1].shape[0]
            expected_loss = -torch.log(torch.tensor(1 / number_of_actions))
            err = torch.abs(initial_loss - expected_loss)
            if err >= self.config["init_loss"]["dev_ratio"] * expected_loss:
                self.error_msg.append(self.main_msgs['poor_init_loss'].format(round((err / expected_loss), 3)))
