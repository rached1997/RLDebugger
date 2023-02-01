import torch.nn
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import get_activation_max_min_bound
from debugger.utils.model_params_getters import is_activation_function


def get_config():
    config = {"Period": 1,
              "start": 10,
              "Dead": {"act_min_thresh": 0.00001, "act_maj_percentile": 95, "neurons_ratio_max_thresh": 0.5},
              "Saturation": {"ro_histo_bins_count": 50, "ro_histo_min": 0.0, "ro_histo_max": 1.0,
                             "ro_max_thresh": 0.85, "neurons_ratio_max_thresh": 0.5},
              "Distribution": {"std_acts_min_thresh": 0.5, "std_acts_max_thresh": 2.0, "f_test_alpha": 0.1},
              "Range": {"disabled": False},
              "Output": {"patience": 5},
              "Numerical_Instability": {"disabled": False}
              }
    return config


class ActivationCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="Activation", config=get_config())
        # self.check_type = "Activation"
        # self.check_period = check_period
        self.nn_data = {}
        self.outputs_metadata = {
            'non_zero_variance': {'patience': 5,
                                  'status': None},
            'max_abs_greater_than_one': {'patience': 5,
                                         'status': None},
            'can_be_negative': {'patience': 5,
                                'status': None}}

    def update_outs_conds(self, outs_array):
        if self.outputs_metadata['non_zero_variance']['status'] is None:
            self.outputs_metadata['non_zero_variance']['status'] = (outs_array.var(dim=0) > 0)
            self.outputs_metadata['max_abs_greater_than_one']['status'] = (torch.abs(outs_array) > 1).any(dim=0)
            self.outputs_metadata['can_be_negative']['status'] = (outs_array < 0).any(dim=0)
        else:
            self.outputs_metadata['non_zero_variance']['status'] |= (outs_array.var(dim=0) > 0)
            self.outputs_metadata['max_abs_greater_than_one']['status'] |= (torch.abs(outs_array) > 1).any(dim=0)
            self.outputs_metadata['can_be_negative']['status'] |= (outs_array < 0).any(dim=0)

    def check_outputs(self, outs_array, error_msg):
        if torch.isinf(outs_array).any():
            error_msg.append(self.main_msgs['out_inf'])
        elif torch.isnan(outs_array).any():
            error_msg.append(self.main_msgs['out_nan'])
            return
        if (self.outputs_metadata['non_zero_variance']['status'] == False).any():
            self.outputs_metadata['non_zero_variance']['patience'] -= 1
            if self.outputs_metadata['non_zero_variance']['patience'] <= 0:
                error_msg.append(self.main_msgs['out_cons'])
        else:
            self.outputs_metadata['non_zero_variance']['patience'] = 5

        if outs_array.shape[1] == 1:
            positive = (outs_array >= 0.).all() and (outs_array <= 1.).all()
            if not (positive):
                error_msg.append(self.main_msgs['output_invalid'])
        else:
            # cannot check sum to 1.0 because of https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
            sum_to_one = (torch.sum(outs_array, dim=1) > 0.95).all() and (torch.sum(outs_array, dim=1) < 1.05).all()
            positive = (outs_array >= 0.).all()
            valid_n_outs = outs_array.shape[1]
            if not (positive and sum_to_one and valid_n_outs):
                error_msg.append(self.main_msgs['output_invalid'])

    def check_activations_range(self, acts_name, acts_array, error_msg):
        if self.config["Range"]["disabled"]: return
        acts_max_bound, acts_min_bound = get_activation_max_min_bound(str(acts_name))
        if (acts_array < acts_max_bound).any():
            main_msg = self.main_msgs['act_ltn'].format(acts_name, acts_min_bound)
            error_msg.append(main_msg)
        if (acts_array > acts_max_bound).any():
            main_msg = self.main_msgs['act_gtn'].format(acts_name, acts_max_bound)
            error_msg.append(main_msg)

    def check_numerical_instabilities(self, acts_name, acts_array, error_msg):
        if self.config["Numerical_Instability"]["disabled"]: return
        if torch.isinf(acts_array).any():
            error_msg.append(self.main_msgs['act_inf'].format(acts_name))
            return True
        if torch.isnan(acts_array).any():
            error_msg.append(self.main_msgs['act_nan'].format(acts_name))
            return True
        return False

    # def check_acts_distribution(self, acts_name, acts_array, is_conv):
    #     acts_array = transform_2d(acts_array, keep='last')
    #     act_std = np.std(acts_array)
    #     if act_std < self.config.dist.std_acts_min_thresh or act_std > self.config.dist.std_acts_max_thresh:
    #         if act_std < self.config.dist.std_acts_min_thresh:
    #             f_test_result = metrics.pure_f_test(acts_array, self.config.dist.std_acts_min_thresh,
    #                                                 self.config.dist.f_test_alpha)
    #         else:
    #             f_test_result = metrics.pure_f_test(acts_array, self.config.dist.std_acts_max_thresh,
    #                                                 self.config.dist.f_test_alpha)
    #         if not (f_test_result[1]):
    #             main_msg = self.main_msgs['conv_act_unstable'] if is_conv else self.main_msgs['fc_act_unstable']
    #             self.react(main_msg.format(acts_name, act_std, self.config.dist.std_acts_min_thresh,
    #                                        self.config.dist.std_acts_max_thresh))

    def update_buffer(self, acts_name, acts_array):
        if acts_name not in self.nn_data.keys():
            self.nn_data[acts_name] = acts_array
            return self.nn_data[acts_name]
        else:
            n = acts_array.shape[0]
            self.nn_data[acts_name][0:-n] = self.nn_data[acts_name][-(self.nn_data.buff_scale - 1) * n:]
            self.nn_data[acts_name][-n:] = acts_array
            return self.nn_data[acts_name]

    def run(self, observations, actions, model, iteration_number):
        activations = {}

        def hook(module, input, output):
            # print(f"{module} activations: {output}")
            activations[module] = output

        def get_activation():
            for name, layer in model.named_modules():
                if is_activation_function(layer):
                    layer.register_forward_hook(hook)

        error_msg = list()
        get_activation()
        outputs = model(torch.tensor(observations))
        self.update_outs_conds(outputs)

        if iteration_number % self.period == 0:
            self.check_outputs(outputs, error_msg)

        for acts_name, acts_array in activations.items():
            acts_buffer = self.update_buffer(acts_name, acts_array)
            if iteration_number < self.config["start"] or iteration_number % self.period != 0:
                continue
            self.check_activations_range(acts_name, acts_buffer, error_msg)
            # if self.check_numerical_instabilities(acts_name, acts_array): continue
            # if self.nn_data.model.act_fn_name in ['sigmoid', 'tanh']:
            #     self.check_saturated_layers(acts_name, acts_buffer, is_conv)
            # else:
            #     self.check_dead_layers(acts_name, acts_buffer, is_conv)
            # self.check_acts_distribution(acts_name, acts_buffer, is_conv)

        return error_msg
