import numpy as np
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss
from debugger.utils.utils import smoothness


def get_config():
    config = {
              "Period": 3,
              "numeric_ins": {"disabled": False},
              "non_dec": {"disabled": False, "window_size": 5, "decr_percentage": 0.05},
              "div": {"disabled": False, "incr_abs_rate_max_thresh": 2},
              "fluct": {"disabled": False, "window_size": 50, "smoothness_ratio_min_thresh": 0.5}
              }
    return config


class OnTrainLossCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="OnTrainLoss", config=get_config())
        self.min_loss = np.inf
        self.current_losses = []
        self.average_losses = []

    def run(self, labels, predictions, loss_fn):
        loss_val = float(get_loss(predictions, labels, loss_fn))
        if self.check_numerical_instabilities(loss_val):
            return
        self.current_losses += [loss_val]
        if self.check_period():
            losses = self.update_losses(np.mean(self.current_losses))
            self.check_loss_curve(losses)
            self.current_losses = []

    def update_losses(self, curr_loss):
        self.min_loss = min(curr_loss, self.min_loss)
        self.average_losses += [curr_loss]
        return np.array(self.average_losses)

    def check_numerical_instabilities(self, loss_value):
        if self.config['numeric_ins']['disabled']:
            return
        if np.isnan(loss_value):
            self.error_msg.append(self.main_msgs['nan_loss'])
            return True
        if np.isinf(loss_value):
            self.error_msg.append(self.main_msgs['inf_loss'])
            return True
        return False

    def check_loss_curve(self, losses):
        n_losses = len(losses)
        if n_losses >= self.config['non_dec']['window_size']:
            dec_pers = np.array(
                [(losses[-i - 1] - losses[-i]) / losses[-i - 1] for i in
                 range(1, self.config['non_dec']['window_size'])])
            if (dec_pers < self.config['non_dec']['decr_percentage']).all() and not (
                    self.config['non_dec']['disabled']):
                self.error_msg.append(self.main_msgs['stagnated_loss'])
        if n_losses >= self.config['non_dec']['window_size']:
            abs_loss_incrs = [losses[n_losses - i] / self.min_loss for i in range(self.config['non_dec']['window_size'],
                                                                                  0, -1)]
            inc_rates = np.array(
                [abs_loss_incrs[-i] / abs_loss_incrs[-i - 1] for i in
                 range(1, self.config['non_dec']['window_size'])])
            if (inc_rates >= self.config['div']['incr_abs_rate_max_thresh']).all() and not (
                    self.config['div']['disabled']):
                self.error_msg.append(self.main_msgs['div_loss'].format(max(inc_rates)))
        if n_losses >= self.config['fluct']['window_size']:
            smoothness_val = smoothness(losses[-self.config['fluct']['window_size']:])
            if smoothness_val < self.config['fluct']['smoothness_ratio_min_thresh'] and not (
                    self.config['fluct']['disabled']):
                self.error_msg.append(self.main_msgs['fluctuated_loss'].format(smoothness_val,
                                                                               self.config['fluct'][
                                                                                   'smoothness_ratio_min_thresh']))
