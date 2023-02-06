from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_model_weights_and_biases
import torch
import numpy as np


def get_config():
    config = {"start": 3,
              "Period": 3,
              "numeric_ins": {"disabled": False},
              "div": {"disabled": False, "window_size": 5, "mav_max_thresh": 100000000, "inc_rate_max_thresh": 2}
              }
    return config


class OnTrainBiasCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="OnTrainBias", config=get_config())
        self.b_reductions = dict()

    def run(self, model):
        _, biases = get_model_weights_and_biases(model)
        for b_name, b_array in biases.items():
            if self.check_numerical_instabilities(b_name, b_array):
                continue

            b_reductions = self.update_b_reductions(b_name, b_array)
            if self.iter_num < self.config['start'] or self.check_period():
                self.check_divergence(b_name, b_reductions)

    def update_b_reductions(self, bias_name, bias_array):
        if bias_name not in self.b_reductions:
            self.b_reductions[bias_name] = []
        self.b_reductions[bias_name].append(torch.mean(torch.abs(bias_array)))
        return self.b_reductions[bias_name]

    def check_numerical_instabilities(self, bias_name, bias_array):
        if self.config['numeric_ins']['disabled']:
            return
        if torch.isinf(bias_array).any():
            self.error_msg.append(self.main_msgs['b_inf'].format(bias_name))
            return True
        if torch.isnan(bias_array).any():
            self.error_msg.append(self.main_msgs['b_nan'].format(bias_name))
            return True
        return False

    def check_divergence(self, bias_name, bias_reductions):
        if self.config['div']['disabled']:
            return
        if bias_reductions[-1] > self.config['div']['mav_max_thresh']:
            self.error_msg.append(self.main_msgs['b_div_1'].format(bias_name, bias_reductions[-1],
                                                                   self.config.div.mav_max_thresh))
        elif len(bias_reductions) >= self.config['div']['window_size']:
            inc_rates = np.array(
                [bias_reductions[-i].cpu().numpy() / bias_reductions[-i - 1].cpu().numpy() for i in
                 range(1, self.config['div']['window_size'])])
            if (inc_rates >= self.config['div']['inc_rate_max_thresh']).all():
                self.error_msg.append(self.main_msgs['b_div_2'].format(bias_name, max(inc_rates),
                                                                       self.config['div']['inc_rate_max_thresh']))
