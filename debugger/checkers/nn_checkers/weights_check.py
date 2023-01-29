import torch.nn
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.metrics import almost_equal, pure_f_test
from debugger.utils.model_params_getters import get_model_layer_names, get_model_weights_and_biases


def get_config():
    config = {
        "Period": 0,
        "Initial_Weight": {
            "disabled": False,
            "f_test_alpha": 0.1
        }
    }
    return config


class WeightsCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type = "Weight", config= get_config())

    def run(self, model):
        error_msg = list()
        initial_weights, _ = get_model_weights_and_biases(model)
        layer_names = get_model_layer_names(model)

        for layer_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1:
                continue
            if almost_equal(torch.var(weight_array), 0.0, rtol=1e-8):
                error_msg.append(self.main_msgs['poor_init'].format(layer_name))
            else:
                if len(shape) == 2:
                    fan_in = shape[0]
                    fan_out = shape[1]
                else:
                    receptive_field_size = torch.prod(torch.tensor(shape[:-2]))
                    fan_in = shape[-2] * receptive_field_size
                    fan_out = shape[-1] * receptive_field_size
                lecun_F, lecun_test = pure_f_test(weight_array, torch.sqrt(1.0 / fan_in),
                                                          self.config["Initial_Weight"]["f_test_alpha"])
                he_F, he_test = pure_f_test(weight_array, torch.sqrt(2.0 / fan_in),
                                                    self.config["Initial_Weight"]["f_test_alpha"])
                glorot_F, glorot_test = pure_f_test(weight_array, torch.sqrt(2.0 / (fan_in + fan_out)),
                                                            self.config["Initial_Weight"]["f_test_alpha"])

                # The following checks can't be done on the last layer
                if layer_name == next(reversed(layer_names.items()))[0]:
                    break
                activation_layer = list(layer_names)[list(layer_names.keys()).index(layer_name) + 1]

                if isinstance(layer_names[activation_layer], torch.nn.ReLU) and not he_test:
                    abs_std_err = torch.abs(torch.std(weight_array) - torch.sqrt(1.0 / fan_in))
                    error_msg.append(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = torch.abs(torch.std(weight_array) - torch.sqrt(2.0 / fan_in))
                    error_msg.append(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = torch.abs(torch.std(weight_array) - torch.sqrt(2.0 / (fan_in + fan_out)))
                    error_msg.append(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    error_msg.append(self.main_msgs['need_init_well'].format(layer_name))

        return error_msg
