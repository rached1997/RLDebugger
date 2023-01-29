import torch.nn
import numpy as np

from debugger.debugger_interface import DebuggerInterface
from debugger.utils.metrics import almost_equal, is_non_2d
from debugger.utils import metrics
from debugger.utils.model_params_getters import get_model_layer_names, get_model_weights_and_biases


class WeightsCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Weight"
        self.check_period = check_period

    def run(self, model):
        error_msg = list()
        initial_weights, _ = get_model_weights_and_biases(model)
        layer_names = get_model_layer_names(model)

        for layer_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1:
                continue
            if almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
                error_msg.append(self.main_msgs['poor_init'].format(layer_name))
            else:
                if len(shape) == 2:
                    fan_in = shape[0]
                    fan_out = shape[1]
                else:
                    receptive_field_size = np.prod(shape[:-2])
                    fan_in = shape[-2] * receptive_field_size
                    fan_out = shape[-1] * receptive_field_size
                lecun_F, lecun_test = metrics.pure_f_test(weight_array, np.sqrt(1.0 / fan_in),
                                                          self.config["Initial_Weight"]["f_test_alpha"])
                he_F, he_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / fan_in),
                                                    self.config["Initial_Weight"]["f_test_alpha"])
                glorot_F, glorot_test = metrics.pure_f_test(weight_array, np.sqrt(2.0 / (fan_in + fan_out)),
                                                            self.config["Initial_Weight"]["f_test_alpha"])

                # The following checks can't be done on the last layer
                if layer_name == next(reversed(layer_names.items()))[0]:
                    break
                activation_layer = list(layer_names)[list(layer_names.keys()).index(layer_name) + 1]

                if isinstance(layer_names[activation_layer], torch.nn.ReLU) and not he_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(1.0 / fan_in))
                    error_msg.append(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / fan_in))
                    error_msg.append(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = np.abs(np.std(weight_array) - np.sqrt(2.0 / (fan_in + fan_out)))
                    error_msg.append(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    error_msg.append(self.main_msgs['need_init_well'].format(layer_name))

        return error_msg


class OverfitWeightsCheck(DebuggerInterface):

    # TODO: Fix the periodicity to make this check runs periodically
    # TODO: Fix the name of this check: find a better idea to group weight Checks

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "OverfitWeight"
        self.check_period = check_period
        self.error_msg = list()
        self.w_reductions = dict()

    def run(self, model):
        weights, _ = get_model_weights_and_biases(model)
        weights = {k: (v, is_non_2d(v)) for k, v in weights.items()}
        for w_name, (w_array, is_conv) in weights.items():
            if self.check_numerical_instabilities(w_name, w_array):
                continue
            w_reductions = self.update_b_reductions(w_name, w_array)
            self.check_sign(w_name, w_array, is_conv)
            self.check_dead(w_name, w_array, is_conv)
            self.check_divergence(w_name, w_reductions, is_conv)

        return self.error_msg

    def update_b_reductions(self, weight_name, weight_array):
        if weight_name not in self.w_reductions:
            self.w_reductions[weight_name] = []
        self.w_reductions[weight_name].append(np.mean(np.abs(weight_array)))
        return self.w_reductions[weight_name]

    def check_numerical_instabilities(self, weight_name, weight_array):
        if self.config['numeric_ins']['disabled']:
            return
        if np.isinf(weight_array).any():
            self.error_msg.append(self.main_msgs['w_inf'].format(weight_name))
            return True
        if np.isnan(weight_array).any():
            self.error_msg.append(self.main_msgs['w_nan'].format(weight_name))
            return True
        return False

    def check_sign(self, weight_name, weight_array, is_conv):
        if self.config['neg']['disabled']:
            return
        neg_ratio = np.count_nonzero(weight_array < 0.) / weight_array.size
        if neg_ratio > self.config['neg']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_sign'] if is_conv else self.main_msgs['fc_w_sign']
            self.error_msg.append(main_msg.format(weight_name, neg_ratio, self.config['neg']['ratio_max_thresh']))

    def check_dead(self, weight_name, weight_array, is_conv):
        if self.config['dead']['disabled']:
            return
        dead_ratio = np.count_nonzero(np.abs(weight_array) < self.config['dead']['val_min_thresh']) / weight_array.size
        if dead_ratio > self.config['dead']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_dead'] if is_conv else self.main_msgs['fc_w_dead']
            self.error_msg.append(main_msg.format(weight_name, dead_ratio, self.config['dead']['val_min_thresh']))

    def check_divergence(self, weight_name, weight_reductions, is_conv):
        if self.config['div']['disabled']:
            return
        if weight_reductions[-1] > self.config['div']['mav_max_thresh']:
            main_msg = self.main_msgs['conv_w_div_1'] if is_conv else self.main_msgs['fc_w_div_1']
            self.error_msg.append(main_msg.format(weight_name, weight_reductions[-1],
                                                  self.config['div']['mav_max_thresh']))
        elif len(weight_reductions) >= self.config['div']['window_size']:
            inc_rates = np.array(
                [weight_reductions[-i] / weight_reductions[-i - 1] for i in range(1, self.config['div']['window_size'])])
            if (inc_rates >= self.config['div']['inc_rate_max_thresh']).all():
                main_msg = self.main_msgs['conv_w_div_2'] if is_conv else self.main_msgs['fc_w_div_2']
                self.error_msg.append(main_msg.format(weight_name, max(inc_rates),
                                                      self.config['div']['inc_rate_max_thresh']))

