import re

import torch.nn
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import get_activation_max_min_bound, transform_2d, compute_ro_B, pure_f_test
from debugger.utils.model_params_getters import is_activation_function
import numpy as np
import torch


# TODO check the configs before the demo
# TODO check this function
def get_config():
    config = {"buff_scale": 10,
              "Period": 10,
              "start": 10,
              "Dead": {"disabled": False, "act_min_thresh": 0.00001, "act_maj_percentile": 95,
                       "neurons_ratio_max_thresh": 0.5},
              "Saturation": {"disabled": False, "ro_histo_bins_count": 50, "ro_histo_min": 0.0, "ro_histo_max": 1.0,
                             "ro_max_thresh": 0.85, "neurons_ratio_max_thresh": 0.5},
              "Distribution": {"disabled": False, "std_acts_min_thresh": 0.5, "std_acts_max_thresh": 2.0,
                               "f_test_alpha": 0.1},
              "Range": {"disabled": False},
              "Output": {"patience": 5},
              "Numerical_Instability": {"disabled": False}
              }
    return config


class OnTrainActivationCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="OnTrainActivation", config=get_config())
        self.acts_data = {}
        self.outputs_metadata = {
            'non_zero_variance': {'status': None},
            'max_abs_greater_than_one': {'status': None},
            'can_be_negative': {'status': None}}

    def update_outs_conds(self, outs_array):
        if self.outputs_metadata['non_zero_variance']['status'] is None:
            self.outputs_metadata['non_zero_variance']['status'] = (outs_array.var(dim=0) > 0)
            self.outputs_metadata['max_abs_greater_than_one']['status'] = (torch.abs(outs_array) > 1).any(dim=0)
            self.outputs_metadata['can_be_negative']['status'] = (outs_array < 0).any(dim=0)
        else:
            self.outputs_metadata['non_zero_variance']['status'] |= (outs_array.var(dim=0) > 0)
            self.outputs_metadata['max_abs_greater_than_one']['status'] |= (torch.abs(outs_array) > 1).any(dim=0)
            self.outputs_metadata['can_be_negative']['status'] |= (outs_array < 0).any(dim=0)

    def check_outputs(self, outs_array):
        if torch.isinf(outs_array).any():
            self.error_msg.append(self.main_msgs['out_inf'])
            return
        elif torch.isnan(outs_array).any():
            self.error_msg.append(self.main_msgs['out_nan'])
            return
        if (self.outputs_metadata['non_zero_variance']['status'] == False).any():
            self.config['patience'] -= 1
            if self.config['patience']['patience'] <= 0:
                self.error_msg.append(self.main_msgs['out_cons'])
        else:
            self.config['patience'] = self.config['Output']['patience']

        if outs_array.shape[1] == 1:
            positive = (outs_array >= 0.).all() and (outs_array <= 1.).all()
            if not positive:
                self.error_msg.append(self.main_msgs['output_invalid'])
        else:
            # cannot check sum to 1.0 because of https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
            sum_to_one = (torch.sum(outs_array, dim=1) > 0.95).all() and (torch.sum(outs_array, dim=1) < 1.05).all()
            positive = (outs_array >= 0.).all()
            valid_n_outs = outs_array.shape[1]
            if not (positive and sum_to_one and valid_n_outs):
                self.error_msg.append(self.main_msgs['output_invalid'])

    def check_activations_range(self, acts_name, acts_array, ):
        if self.config["Range"]["disabled"]:
            return
        acts_max_bound, acts_min_bound = get_activation_max_min_bound(str(acts_name))
        if (acts_array < acts_min_bound).any():
            main_msg = self.main_msgs['act_ltn'].format(acts_name, acts_min_bound)
            self.error_msg.append(main_msg)
        if (acts_array > acts_max_bound).any():
            main_msg = self.main_msgs['act_gtn'].format(acts_name, acts_max_bound)
            self.error_msg.append(main_msg)

    def check_numerical_instabilities(self, acts_name, acts_array):
        if self.config["Numerical_Instability"]["disabled"]:
            return
        if torch.isinf(acts_array).any():
            self.error_msg.append(self.main_msgs['act_inf'].format(acts_name))
            return True
        if torch.isnan(acts_array).any():
            self.error_msg.append(self.main_msgs['act_nan'].format(acts_name))
            return True
        return False

    def check_saturated_layers(self, acts_name, acts_array):
        if self.config['Saturation']['disabled']: return
        acts_array = transform_2d(acts_array, keep='last').numpy()
        ro_Bs = np.apply_along_axis(compute_ro_B, 0, acts_array, min_out=self.config["Saturation"]['ro_histo_min'],
                                    max_out=self.config["Saturation"]['ro_histo_max'],
                                    bins_count=self.config["Saturation"]["ro_histo_bins_count"])
        saturated_count = np.count_nonzero(ro_Bs > self.config["Saturation"]['ro_max_thresh'])
        saturated_ratio = saturated_count / ro_Bs.shape[0]
        if saturated_ratio > self.config.sat.neurons_ratio_max_thresh:
            main_msg = self.main_msgs['act_sat']
            self.error_msg.append(main_msg.format(saturated_count, ro_Bs.size, acts_name))

    def check_dead_layers(self, acts_name, acts_array):
        if self.config["Dead"]["disabled"]:
            return
        acts_array = transform_2d(acts_array, keep='last')
        major_values = np.percentile(np.abs(acts_array), q=self.config["Dead"]["act_maj_percentile"], axis=0)
        dead_count = np.count_nonzero(major_values < self.config["Dead"]["act_min_thresh"])
        dead_ratio = dead_count / major_values.shape[0]
        if dead_ratio > self.config["Dead"]["neurons_ratio_max_thresh"]:
            main_msg = self.main_msgs['act_dead']
            self.error_msg.append(main_msg.format(dead_count, major_values.size, acts_name))

    def check_acts_distribution(self, acts_name, acts_array):
        if self.config["Distribution"]["disabled"]:
            return
        acts_array = transform_2d(acts_array, keep='last')
        act_std = np.std(acts_array)
        if act_std < self.config["Distribution"]["std_acts_min_thresh"] \
                or act_std > self.config["Distribution"]["std_acts_max_thresh"]:
            if act_std < self.config["Distribution"]["std_acts_min_thresh"]:
                f_test_result = pure_f_test(acts_array, self.config["Distribution"]["std_acts_min_thresh"],
                                            self.config["Distribution"]["f_test_alpha"])
            else:
                f_test_result = pure_f_test(acts_array, self.config["Distribution"]["std_acts_max_thresh"],
                                            self.config["Distribution"]["f_test_alpha"])
            if not (f_test_result[1]):
                main_msg = self.main_msgs['act_unstable']
                self.error_msg.append(
                    main_msg.format(acts_name, act_std, self.config["Distribution"]["std_acts_min_thresh"],
                                    self.config["Distribution"]["std_acts_max_thresh"]))

    def update_buffer(self, acts_name, acts_array):
            n = acts_array.shape[0]
            self.acts_data[acts_name][0:-n] = self.acts_data[acts_name][-(self.config['buff_scale'] - 1) * n:]
            self.acts_data[acts_name][-n:] = acts_array.cpu().detach().numpy()
            return self.acts_data[acts_name]

    def set_acts_data(self, observations_size, activations):
        acts_data = {}
        for i, (acts_name, acts_array) in enumerate(activations.items()):
            acts_name = re.sub(r'\([^()]*\)', '', str(acts_name)) + "_" + str(i)
            dims = [int(dim) for dim in acts_array.shape[1:]]
            buffer_size = self.config['buff_scale'] * observations_size
            acts_data[acts_name] = np.zeros(shape=(buffer_size, *dims))
        self.acts_data = acts_data

    def run(self, observations, model):
        activations = {}

        def hook(module, input, output):
            activations[module] = output

        def get_activation():
            for name, layer in model.named_modules():
                if is_activation_function(layer):
                    layer.register_forward_hook(hook)

        get_activation()
        outputs = model(observations)
        if self.acts_data == {}:
            self.set_acts_data(observations.shape[0], activations)

        self.update_outs_conds(outputs)

        if self.iter_num % self.period == 0:
            self.check_outputs(outputs)

        for i, (acts_name, acts_array) in enumerate(activations.items()):
            acts_name = re.sub(r'\([^()]*\)', '', str(acts_name)) + "_" + str(i)
            acts_buffer = self.update_buffer(acts_name, acts_array)
            if self.iter_num < self.config["start"] or self.iter_num % self.period != 0:
                continue
            self.check_activations_range(acts_name, acts_buffer)
            if self.check_numerical_instabilities(acts_name, acts_array): continue
            if acts_name in ['Sigmoid', 'Tanh']:
                self.check_saturated_layers(acts_name, acts_buffer)
            else:
                self.check_dead_layers(acts_name, acts_buffer)

            self.check_acts_distribution(acts_name, acts_buffer)
