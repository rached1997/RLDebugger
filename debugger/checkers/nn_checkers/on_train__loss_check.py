import numpy as np
import torch
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss
from debugger.utils.utils import smoothness


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
              "Period": 100,
              "numeric_ins": {"disabled": False},
              "non_dec": {"disabled": False, "window_size": 5, "decr_percentage": 0.05},
              "div": {"disabled": False, "incr_abs_rate_max_thresh": 2, "window_size": 5},
              "fluct": {"disabled": False, "window_size": 50, "smoothness_ratio_min_thresh": 0.5}
              }
    return config


class OnTrainLossCheck(DebuggerInterface):
    """
    The check is in charge of verifying the loss function during training.
    """

    def __init__(self):
        super().__init__(check_type="OnTrainLoss", config=get_config())
        self.min_loss = np.inf
        self.current_losses = []
        self.average_losses = []

    def run(self, targets: torch.Tensor, predictions: torch.Tensor, loss_fn: torch.nn.Module) -> None:
        """
        This function performs multiple checks on the loss function during the training:

        (1) Check the numerical instabilities of loss values. (check the function check_numerical_instabilities
        for more details)
        (2) Check the abnormal loss curvature of the loss (check the function check_loss_curve for more details)

        Args:
        targets (Tensor): A sample of targets collected periodically during the training.
        predictions (Tensor): A sample of predictions collected periodically during the training.
        loss_fn (torch.nn.Module): the loss function of the model.
        """
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
