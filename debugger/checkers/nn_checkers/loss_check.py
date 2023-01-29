import numpy as np
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
from debugger.utils.utils import numpify
from debugger.utils.metrics import smoothness


class LossCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Loss"
        self.check_period = check_period

    def run(self, labels, predictions, loss, model):
        error_msgs = list()
        losses = []
        n = self.config["init_loss"]["size_growth_rate"]
        while n <= (self.config["init_loss"]["size_growth_rate"] * self.config["init_loss"]["size_growth_iters"]):
            derived_batch_y = np.concatenate(n * [labels], axis=0)
            derived_predictions = np.concatenate(n * [numpify(predictions)], axis=0)
            loss_value = float(get_loss(derived_predictions, derived_batch_y, loss))
            losses.append(loss_value)
            n *= self.config["init_loss"]["size_growth_rate"]
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config["init_loss"]["size_growth_rate"]) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            error_msgs.append(self.main_msgs['poor_reduction_loss'])

        initial_loss = float(get_loss(predictions, labels, loss))
        # specify here the number of actions
        initial_weights, _ = get_model_weights_and_biases(model)
        number_of_actions = list(initial_weights.items())[-1][1].shape[0]
        expected_loss = -np.log(1 / number_of_actions)
        err = np.abs(initial_loss - expected_loss)
        if err >= self.config["init_loss"]["dev_ratio"] * expected_loss:
            error_msgs.append(self.main_msgs['poor_init_loss'].format(round((err / expected_loss), 3)))
        return error_msgs


class OverfitLossCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "OverfitLoss"
        self.check_period = check_period
        self.step_losses = []
        self.error_msg = list()
        self.min_loss = np.inf

    def run(self, labels, predictions, loss):
        loss_val = float(get_loss(predictions, labels, loss))
        if self.check_numerical_instabilities(loss_val):
            return
        losses = self.update_losses(loss_val)
        self.check_loss_curve(losses)
        return self.error_msg

    def update_losses(self, curr_loss):
        self.min_loss = min(curr_loss, self.min_loss)
        self.step_losses += [curr_loss]
        return self.step_losses

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
                [abs_loss_incrs[-i] / abs_loss_incrs[-i - 1] for i in range(1, self.config['non_dec']['window_size'])])
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