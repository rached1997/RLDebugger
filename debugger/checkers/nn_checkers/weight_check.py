import torch.nn
import numpy as np

from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import is_non_2d, pure_f_test, almost_equal
from debugger.utils.model_params_getters import get_model_weights_and_biases


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "start": 100,
        "Period": 100,
        "numeric_ins": {"disabled": False},
        "neg": {"disabled": False, "ratio_max_thresh": 0.95},
        "dead": {"disabled": False, "val_min_thresh": 0.00001, "ratio_max_thresh": 0.95},
        "div": {"disabled": False, "window_size": 5, "mav_max_thresh": 100000000, "inc_rate_max_thresh": 2},
        "Initial_Weight": {"disabled": False,"f_test_alpha": 0.1}
    }
    return config


class WeightsCheck(DebuggerInterface):
    """
        The check in charge of verifying the weight values during training.
    """
    def __init__(self):
        super().__init__(check_type="Weight", config=get_config())
        self.w_reductions = dict()

    def run(self, model: torch.nn.Module) -> None:
        """
        This function performs multiple checks on the weight during the training:

        (1) run the pre-checks described in the function run_pre_checks
        (2) Check the numerical instabilities of weight values (check the function check_numerical_instabilities
        for more details)
        (3) Check the divergence of weight values (check the function check_divergence for more details)

        Args:
            model (torch.nn.Module): The model being trained

        Returns:
            None
        """
        if self.iter_num == 1:
            self.run_pre_checks(model)
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

    def update_w_reductions(self, weight_name: str, weight_array: torch.Tensor) -> torch.Tensor:
        """
        Updates and save the weights periodically. At each step, the mean value of the weights is stored.

        Args:
            weight_name (str): The name of the layer to be validated.
            weight_array (Tensor): The weights obtained from the specified layer.

        Returns:
            None
        """
        if weight_name not in self.w_reductions:
            self.w_reductions[weight_name] = []
        self.w_reductions[weight_name].append(torch.mean(torch.abs(weight_array)))
        return self.w_reductions[weight_name]

    def check_numerical_instabilities(self, weight_name: str, weight_array: torch.Tensor) -> bool:
        """
        Validates the numerical stability of bias values during training.

        Args:
            weight_name (str): The name of the layer to be validated.
            weight_array (torch.Tensor): The weights obtained from the specified layer.

        Returns:
            (bool)
        """
        if self.config['numeric_ins']['disabled']:
            return False
        if torch.isinf(weight_array).any():
            self.error_msg.append(self.main_msgs['w_inf'].format(weight_name))
            return True
        if torch.isnan(weight_array).any():
            self.error_msg.append(self.main_msgs['w_nan'].format(weight_name))
            return True
        return False

    def check_sign(self, weight_name: str, weight_array: torch.Tensor, is_conv: bool) -> None:
        """
        This function check Over-Negative weight in each layer. A layer’s weights are considered over-negative,
        when, the ratio of negative values in the tensor elements is very high. This state of weights are likely
        to be problematic for the learning dynamics. More details on theoretical proof of this function can be found
        here:
            - https://arxiv.org/abs/1806.06068

        Args:
            weight_name (str): The name of the layer to be validated.
            weight_array (Tensor): The weights obtained from the specified layer.
            is_conv (bool): a boolean indicating whether the current layer is a Conv layer.

        Returns:
            None
        """
        if self.config['neg']['disabled']:
            return
        neg_ratio = (weight_array < 0.).sum().item() / torch.numel(weight_array)
        if neg_ratio > self.config['neg']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_sign'] if is_conv else self.main_msgs['fc_w_sign']
            self.error_msg.append(main_msg.format(weight_name, neg_ratio, self.config['neg']['ratio_max_thresh']))

    def check_dead(self, weight_name: str, weight_array: torch.Tensor, is_conv: bool) -> None:
        """
        This function check Dead weight in each layer. A layer’s weights are considered dead, when, the ratio of zeros
        values in the tensor elements is very high. This state of weights are likely to be problematic for the learning
        dynamics.More details on theoretical proof of this function can be found here:
            - https://arxiv.org/abs/1806.06068

        Args:
            weight_name (str): The name of the layer to be validated.
            weight_array (Tensor): The weights obtained from the specified layer.
            is_conv: (bool): a boolean indicating whether the current layer is a Conv layer.

        Returns:

        """
        if self.config['dead']['disabled']:
            return
        dead_ratio = torch.sum(
            (torch.abs(weight_array) < self.config['dead']['val_min_thresh']).int()).item() / torch.numel(weight_array)
        if dead_ratio > self.config['dead']['ratio_max_thresh']:
            main_msg = self.main_msgs['conv_w_dead'] if is_conv else self.main_msgs['fc_w_dead']
            self.error_msg.append(main_msg.format(weight_name, dead_ratio, self.config['dead']['val_min_thresh']))

    def check_divergence(self, weight_name: str, weight_reductions, is_conv: bool):
        """
        This function check weight divergence, as weights risk divergence, and may go towards inf.
        High initial weights or learning rate coupled with a lack of or inadequate regularization results in rapidly
        growing weight updates, resulting to increasing values until they hit inf.
        This function automates a verification routine that watches continuously the absolute averages of weights
         are not diverging. More details on theoretical proof of this function can be found here:
            - https://arxiv.org/pdf/2204.00694.pdf

        Args:
            weight_name: (str) The name of the layer to be validated.
            weight_reductions: (Tensor): The weights obtained from the specified layer.
            is_conv: (bool) a boolean indicating whether or not the current layer is a Conv layer.

        Returns:
            None
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

    def run_pre_checks(self, model):
        """
        Perform multiple checks on the initial values of the weights before training. The checks include:

         1. Confirming if there is substantial differences between parameter values by computing their variance and
        verifying it is not equal to zero.

        2. Ensuring the distribution of initial random values matches the recommended distribution for the chosen
        activation function. This is done by comparing the variance of weights with the recommended variance,
        using the f-test. The recommended variances for different activation layers are:
            A. Lecun initialization for sigmoid activation. (check this paper for more details
                http://yann.lecun.org/exdb/publis/pdf/lecun-98b.pdf )
            B. Glorot initialization for tanh activation (check this paper for more details
                https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf )
            C. He initialization for ReLU activation. (check this paper for more details
                https://arxiv.org/pdf/1502.01852.pdf )

        Args:
            model (nn.Module): The model to be trained.

        """
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
                    abs_std_err = torch.abs(torch.std(weight_array) - np.sqrt((1.0 / fan_in)))
                    self.error_msg.append(self.main_msgs['need_he'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Tanh) and not glorot_test:
                    abs_std_err = torch.abs(torch.std(weight_array) - np.sqrt((2.0 / fan_in)))
                    self.error_msg.append(self.main_msgs['need_glorot'].format(layer_name, abs_std_err))
                elif isinstance(layer_names[activation_layer], torch.nn.Sigmoid) and not lecun_test:
                    abs_std_err = torch.abs(torch.std(weight_array) - np.sqrt((2.0 / (fan_in + fan_out))))
                    self.error_msg.append(self.main_msgs['need_lecun'].format(layer_name, abs_std_err))
                elif not (lecun_test or he_test or glorot_test):
                    self.error_msg.append(self.main_msgs['need_init_well'].format(layer_name))

    def compute_f_test(self, weight_array):
        """
        This function compute the f-test [1] to verify the equality between the actual variance of each weight and
        its recommended variance given the input size.
            - [1] : (https://history.wisc.edu/publications/correlation-and-regression-analysis-a-historians-guide/)

        Args:
            weight_array: (Tensor) The weights obtained from the specified layer.

        Returns:

        """
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight_array)
        lecun_F, lecun_test = pure_f_test(weight_array, np.sqrt((1.0 / fan_in)),
                                          self.config["Initial_Weight"]["f_test_alpha"])
        he_F, he_test = pure_f_test(weight_array, np.sqrt((2.0 / fan_in)),
                                    self.config["Initial_Weight"]["f_test_alpha"])
        glorot_F, glorot_test = pure_f_test(weight_array, np.sqrt((2.0 / (fan_in + fan_out))),
                                            self.config["Initial_Weight"]["f_test_alpha"])

        return lecun_test, he_test, glorot_test, fan_in, fan_out