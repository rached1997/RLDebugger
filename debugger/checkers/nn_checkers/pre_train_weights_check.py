import torch.nn
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import almost_equal, pure_f_test
from debugger.utils.model_params_getters import get_model_weights_and_biases
import numpy as np


def get_config():
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """

    config = {
        "Period": 0,
        "Initial_Weight": {
            "disabled": False,
            "f_test_alpha": 0.1
        }
    }
    return config


class PreTrainWeightsCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="PreTrainWeight", config=get_config())

    def run(self, model):
        """
        Perform multiple checks on the initial values of the weights before training. The checks include:

         1. Confirming if there is substantial differences between parameter values by computing their variance and
        verifying it is not equal to zero.

        2. Ensuring the distribution of initial random values matches the recommended distribution for the chosen
        activation function. This is done by comparing the variance of weights with the recommended variance,
        using the f-test. The recommended variances for different activation layers are:
            a. Lecun initialization for sigmoid activation. (check this paper for more details
                http://yann.lecun.org/exdb/publis/pdf/lecun-98b.pdf )
            b. Glorot initialization for tanh activation (check this paper for more details
                https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf )
            c. He initialization for ReLU activation. (check this paper for more details
                https://arxiv.org/pdf/1502.01852.pdf )

        Args:
            model (nn.Module): The model to be trained.

        """
        if not self.check_period():
            return
        initial_weights, _ = get_model_weights_and_biases(model)
        layer_names = dict(model.named_modules())
        last_layer_name = list(layer_names.keys())[-1]

        for layer_name, weight_array in initial_weights.items():
            if weight_array.dim() == 1 and weight_array.shape[0] == 1:
                continue
            if almost_equal(torch.var(weight_array), 0.0, rtol=1e-8):
                self.error_msg.append(self.main_msgs['poor_init'].format(layer_name))
            else:

                lecun_test, he_test, glorot_test, fan_in, fan_out = self.compute_f_test(weight_array)

                # The following checks can't be done on the last layer
                if layer_name == last_layer_name:
                    break
                activation_layer = list(layer_names)[list(layer_names.keys()).index(layer_name) + 1]

                if isinstance(layer_names[activation_layer], torch.nn.ReLU) and not he_test:
                    abs_std_err = np.abs(torch.std(weight_array) - np.sqrt((1.0 / fan_in)))
                    self.error_msg.append(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = np.abs(torch.std(weight_array) - np.sqrt((2.0 / fan_in)))
                    self.error_msg.append(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = np.abs(torch.std(weight_array) - np.sqrt((2.0 / (fan_in + fan_out))))
                    self.error_msg.append(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    self.error_msg.append(self.main_msgs['need_init_well'].format(layer_name))

    def compute_f_test(self, weight_array):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight_array)
        lecun_F, lecun_test = pure_f_test(weight_array, np.sqrt((1.0 / fan_in)),
                                          self.config["Initial_Weight"]["f_test_alpha"])
        he_F, he_test = pure_f_test(weight_array, np.sqrt((2.0 / fan_in)),
                                    self.config["Initial_Weight"]["f_test_alpha"])
        glorot_F, glorot_test = pure_f_test(weight_array, np.sqrt((2.0 / (fan_in + fan_out))),
                                            self.config["Initial_Weight"]["f_test_alpha"])

        return lecun_test, he_test, glorot_test, fan_in, fan_out


