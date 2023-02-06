from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_model_weights_and_biases
import torch
import numpy as np


def get_config():
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {"start": 3,
              "Period": 3,
              "numeric_ins": {"disabled": False},
              "div": {"disabled": False, "window_size": 5, "mav_max_thresh": 100000000, "inc_rate_max_thresh": 2}
              }
    return config


class OnTrainBiasCheck(DebuggerInterface):
    """
    The check is in charge of verifying the bias values during training.
    """

    def __init__(self):
        super().__init__(check_type="OnTrainBias", config=get_config())
        self.b_reductions = dict()

    def run(self, model):
        """
        This function performs multiple checks on the bias during the training:

        (1) Check the numerical instabilities of bias values (check the function check_numerical_instabilities
        for more details)
        (2) Check the divergence of bias values (check the function check_divergence for more details)

        Args:
            model: (nn.model): The model being trained

        Returns:
            None
        """
        _, biases = get_model_weights_and_biases(model)
        for b_name, b_array in biases.items():
            if self.check_numerical_instabilities(b_name, b_array):
                continue

            b_reductions = self.update_b_reductions(b_name, b_array)
            if self.iter_num < self.config['start'] or self.check_period():
                self.check_divergence(b_name, b_reductions)

    def update_b_reductions(self, bias_name, bias_array):
        """
        Updates and save the biases periodically. At each step, the mean value of the biases is stored.

        Args:
            # TODO: check this affirmation.
            bias_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'bias'.
            bias_array (Tensor): The biases obtained from the specified layer.

        Returns:
            (Tensor): all average biases obtained during training.
        """
        if bias_name not in self.b_reductions:
            self.b_reductions[bias_name] = []
        self.b_reductions[bias_name].append(torch.mean(torch.abs(bias_array)))
        return self.b_reductions[bias_name]

    def check_numerical_instabilities(self, bias_name, bias_array):
        """
        Validates the numerical stability of bias values during training.

        Args:
            # TODO: check this affirmation.
            bias_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'bias'.
            bias_array: (Tensor): The biases obtained from the specified layer.

        Returns:
            (bool): True if there is any NaN or infinite value present, False otherwise.
        """
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
        """
        This function check bias divergence, as biases risk divergence, and may go towards inf. Biases can become
        huge in cases when features (observation) do not adequately explain the predicted outcome or are ineffective.
        This function automates a verification routine that watches continuously the absolute averages of bias
         are not diverging. More details on theoretical proof of this function can be found here:
            - https://arxiv.org/pdf/2204.00694.pdf

        Args:
            # TODO: check this affirmation.
            bias_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'bias'.
            bias_reductions:

        Returns:

        """
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
