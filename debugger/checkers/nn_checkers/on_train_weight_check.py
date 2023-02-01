import torch.nn
import numpy as np

from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import is_non_2d
from debugger.utils.model_params_getters import get_model_weights_and_biases


def get_config():
    config = {
        "start": 3,
        "Period": 3,
        "numeric_ins": {"disabled": False},
        "neg": {"disabled": False, "ratio_max_thresh": 0.95},
        "dead": {"disabled": False, "val_min_thresh": 0.00001, "ratio_max_thresh": 0.95},
        "div": {"disabled": False, "window_size": 5, "mav_max_thresh": 100000000, "inc_rate_max_thresh": 2}
    }
    return config


class OnTrainWeightsCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainWeight", config=get_config())
        self.w_reductions = dict()

    def run(self, model):
        weights, _ = get_model_weights_and_biases(model)
        weights = {k: (v, is_non_2d(v)) for k, v in weights.items()}
        for w_name, (w_array, is_conv) in weights.items():
            if self.check_numerical_instabilities(w_name, w_array):
                continue
            w_reductions = self.update_b_reductions(w_name, w_array)

            if self.iter_num < self.config['start'] or self.check_period():
                self.check_sign(w_name, w_array, is_conv)
                self.check_dead(w_name, w_array, is_conv)
                self.check_divergence(w_name, w_reductions, is_conv)

    def update_b_reductions(self, weight_name, weight_array):
        if weight_name not in self.w_reductions:
            self.w_reductions[weight_name] = []
        self.w_reductions[weight_name].append(torch.mean(torch.abs(weight_array)))
        return self.w_reductions[weight_name]

    def check_numerical_instabilities(self, weight_name, weight_array):
        if self.config['numeric_ins']['disabled']:
            return
        if torch.isinf(weight_array).any():
            self.error_msg.append(self.main_msgs['w_inf'].format(weight_name))
            return True
        if torch.isnan(weight_array).any():
            self.error_msg.append(self.main_msgs['w_nan'].format(weight_name))
            return True
        return False

    def check_sign(self, weight_name, weight_array, is_conv):
        if self.config['neg']['disabled']:
            return
        neg_ratio = (weight_array < 0.).sum().item() / torch.numel(weight_array)
        if neg_ratio > self.config['neg']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_sign'] if is_conv else self.main_msgs['fc_w_sign']
            self.error_msg.append(main_msg.format(weight_name, neg_ratio, self.config['neg']['ratio_max_thresh']))

    def check_dead(self, weight_name, weight_array, is_conv):
        if self.config['dead']['disabled']:
            return
        dead_ratio = torch.sum(
            (torch.abs(weight_array) < self.config['dead']['val_min_thresh']).int()).item() / torch.numel(weight_array)
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
                [weight_reductions[-i] / weight_reductions[-i - 1] for i in
                 range(1, self.config['div']['window_size'])])
            if (inc_rates >= self.config['div']['inc_rate_max_thresh']).all():
                main_msg = self.main_msgs['conv_w_div_2'] if is_conv else self.main_msgs['fc_w_div_2']
                self.error_msg.append(main_msg.format(weight_name, max(inc_rates),
                                                      self.config['div']['inc_rate_max_thresh']))
