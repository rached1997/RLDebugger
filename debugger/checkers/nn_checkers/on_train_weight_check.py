import torch.nn
import numpy as np

from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import is_non_2d
from debugger.utils.model_params_getters import get_model_weights_and_biases


def get_config():
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
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
    """
        The check in charge of verifying the weight values during training.
    """
    def __init__(self):
        super().__init__(check_type="OnTrainWeight", config=get_config())
        self.w_reductions = dict()

    def run(self, model):
        """
        This function performs multiple checks on the weight during the training:

        (1) Check the numerical instabilities of weight values (check the function check_numerical_instabilities
        for more details)
        (2) Check the divergence of weight values (check the function check_divergence for more details)

        Args:
            model: (nn.model) The model being trained

        Returns:
            None
        """
        weights, _ = get_model_weights_and_biases(model)
        weights = {k: (v, is_non_2d(v)) for k, v in weights.items()}
        for w_name, (w_array, is_conv) in weights.items():
            if self.check_numerical_instabilities(w_name, w_array):
                continue
            w_reductions = self.update_w_reductions(w_name, w_array)

            if self.iter_num < self.config['start'] or self.check_period():
                self.check_sign(w_name, w_array, is_conv)
                self.check_dead(w_name, w_array, is_conv)
                self.check_divergence(w_name, w_reductions, is_conv)

    def update_w_reductions(self, weight_name, weight_array):
        """
        Updates and save the weights periodically. At each step, the mean value of the weights is stored.

        Args:
            # TODO: check this affirmation.
            weight_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'weight'.
            weight_array: (Tensor): The weights obtained from the specified layer.

        Returns:
            None
        """
        if weight_name not in self.w_reductions:
            self.w_reductions[weight_name] = []
        self.w_reductions[weight_name].append(torch.mean(torch.abs(weight_array)))
        return self.w_reductions[weight_name]

    def check_numerical_instabilities(self, weight_name, weight_array):
        """
        Validates the numerical stability of bias values during training.

        Args:
            # TODO: check this affirmation.
            weight_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'weight'.
            weight_array: (Tensor): The weights obtained from the specified layer.

        Returns:
            None
        """
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
        """
        This function check Over-Negative weight in each layer. A layer’s weights are considered over-negative,
        when, the ratio of negative values in the tensor elements is very high. This state of weights are likely
        to be problematic for the learning dynamics. More details on theoretical proof of this function can be found
        here:
            - https://arxiv.org/abs/1806.06068

        Args:
            # TODO: check this affirmation.
            weight_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'weight'.
            weight_array: (Tensor): The weights obtained from the specified layer.
            is_conv: (bool) a boolean indicating whether or not the current layer is a Conv layer.

        Returns:

        """
        if self.config['neg']['disabled']:
            return
        neg_ratio = (weight_array < 0.).sum().item() / torch.numel(weight_array)
        if neg_ratio > self.config['neg']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_sign'] if is_conv else self.main_msgs['fc_w_sign']
            self.error_msg.append(main_msg.format(weight_name, neg_ratio, self.config['neg']['ratio_max_thresh']))

    def check_dead(self, weight_name, weight_array, is_conv):
        """
        This function check Dead weight in each layer. A layer’s weights are considered dead, when, the ratio of zeros
        values in the tensor elements is very high. This state of weights are likely to be problematic for the learning
        dynamics.More details on theoretical proof of this function can be found here:
            - https://arxiv.org/abs/1806.06068

        Args:
            # TODO: check this affirmation.
            weight_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'weight'.
            weight_array: (Tensor): The weights obtained from the specified layer.
            is_conv: (bool) a boolean indicating whether or not the current layer is a Conv layer.

        Returns:

        """
        if self.config['dead']['disabled']:
            return
        dead_ratio = torch.sum(
            (torch.abs(weight_array) < self.config['dead']['val_min_thresh']).int()).item() / torch.numel(weight_array)
        if dead_ratio > self.config['dead']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_dead'] if is_conv else self.main_msgs['fc_w_dead']
            self.error_msg.append(main_msg.format(weight_name, dead_ratio, self.config['dead']['val_min_thresh']))

    def check_divergence(self, weight_name, weight_reductions, is_conv):
        """
        This function check weight divergence, as weights risk divergence, and may go towards inf.
        High initial weights or learning rate coupled with a lack of or inadequate regularization results in rapidly
        growing weight updates, resulting to increasing values until they hit inf.
        This function automates a verification routine that watches continuously the absolute averages of weights
         are not diverging. More details on theoretical proof of this function can be found here:
            - https://arxiv.org/pdf/2204.00694.pdf

        Args:
            # TODO: check this affirmation.
            weight_name: (str) The name of the layer to be validated. The name should include the name of the
            layer followed with a prefix 'weight'.
            weight_reductions: (Tensor): The weights obtained from the specified layer.
            is_conv: (bool) a boolean indicating whether or not the current layer is a Conv layer.

        Returns:

        """
        if self.config['div']['disabled']:
            return
        if weight_reductions[-1] > self.config['div']['mav_max_thresh']:
            main_msg = self.main_msgs['conv_w_div_1'] if is_conv else self.main_msgs['fc_w_div_1']
            self.error_msg.append(main_msg.format(weight_name, weight_reductions[-1],
                                                  self.config['div']['mav_max_thresh']))
        elif len(weight_reductions) >= self.config['div']['window_size']:
            inc_rates = np.array(
                [weight_reductions[-i].cpu().numpy() / weight_reductions[-i - 1].cpu().numpy() for i in
                 range(1, self.config['div']['window_size'])])
            if (inc_rates >= self.config['div']['inc_rate_max_thresh']).all():
                main_msg = self.main_msgs['conv_w_div_2'] if is_conv else self.main_msgs['fc_w_div_2']
                self.error_msg.append(main_msg.format(weight_name, max(inc_rates),
                                                      self.config['div']['inc_rate_max_thresh']))
