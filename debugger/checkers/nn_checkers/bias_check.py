from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_model_weights_and_biases, get_last_layer_activation
import torch
import numpy as np

from debugger.utils.utils import get_balance, get_probas


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {"start": 100,
              "Period": 100,
              "numeric_ins": {"disabled": False},
              "div": {"disabled": False, "window_size": 5, "mav_max_thresh": 100000000, "inc_rate_max_thresh": 2}
              }
    return config


class BiasCheck(DebuggerInterface):
    """
    The check is in charge of verifying the bias values during training.
    """

    def __init__(self):
        super().__init__(check_type="Bias", config=get_config())
        self.b_reductions = dict()

    def run(self, model: torch.nn.Module, training_observations: torch.Tensor = None) -> None:
        """
        This function performs multiple checks on the bias during the training:

        (1) Run the pre-checks described in the function run_pre_checks
        (2) Check the numerical instabilities of bias values (check the function check_numerical_instabilities
        for more details)
        (3) Check the divergence of bias values (check the function check_divergence for more details)

        Args:
            model (nn.model): The model being trained
            training_observations (Tensor): Initial sample of observations.

        Returns:
            None
        """
        if self.iter_num == 1:
            self.run_pre_checks(model, training_observations)
        _, biases = get_model_weights_and_biases(model)
        for b_name, b_array in biases.items():
            if self.check_numerical_instabilities(b_name, b_array):
                continue

            b_reductions = self.update_b_reductions(b_name, b_array)
            if self.iter_num < self.config['start'] or self.check_period():
                self.check_divergence(b_name, b_reductions)

    def update_b_reductions(self, bias_name: str, bias_array: torch.Tensor) -> torch.Tensor:
        """
        Updates and save the biases periodically. At each step, the mean value of the biases is stored.

        Args:
            bias_name (str): The name of the layer to be validated.
            bias_array (Tensor): The biases obtained from the specified layer.

        Returns:
            (Tensor): all average biases obtained during training.
        """
        if bias_name not in self.b_reductions:
            self.b_reductions[bias_name] = []
        self.b_reductions[bias_name].append(torch.mean(torch.abs(bias_array)))
        return self.b_reductions[bias_name]

    def check_numerical_instabilities(self, bias_name: str, bias_array: torch.Tensor) -> bool:
        """
        Validates the numerical stability of bias values during training.

        Args:
            bias_name: (str) The name of the layer to be validated.
            bias_array: (Tensor): The biases obtained from the specified layer.

        Returns:
            (bool): True if there is any NaN or infinite value present, False otherwise.
        """
        if self.config['numeric_ins']['disabled']:
            return False
        if torch.isinf(bias_array).any():
            self.error_msg.append(self.main_msgs['b_inf'].format(bias_name))
            return True
        if torch.isnan(bias_array).any():
            self.error_msg.append(self.main_msgs['b_nan'].format(bias_name))
            return True
        return False

    def check_divergence(self, bias_name: str, bias_reductions: torch.Tensor) -> None:
        """
        This function check bias divergence, as biases risk divergence, and may go towards inf. Biases can become
        huge in cases when features (observation) do not adequately explain the predicted outcome or are ineffective.
        This function automates a verification routine that watches continuously the absolute averages of bias
         are not diverging. More details on theoretical proof of this function can be found here:
            - https://arxiv.org/pdf/2204.00694.pdf

        Args:
            bias_name: (str) The name of the layer to be validated.
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

    def run_pre_checks(self, model, training_observations) -> None:
        """
        This function performs multiple checks on the bias initial values of the model:

        (1) Verifies the existence of the bias
        (2) Checks if the bias of the last layer is non-zero when the model's output in the initial observation set
        is imbalanced.
        (3) Validates if the bias of the last layer matches the label ratio when the output of the model in the
        initial observation set is imbalanced, using the formula bi = log(pi / (1-pi)), where pi is the proportion of
        observations of the label (actions) corresponding to the bias bi of unit i.
        (4) Confirms that the bias is not set to zero.

        Args:
        model (torch.nn.Module): The model that is being trained.
        observations (torch.Tensor): A sample of observations collected before the start of the training process.
        """
        _, initial_biases = get_model_weights_and_biases(model)
        if not initial_biases:
            self.error_msg.append(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(torch.sum(b_array) == 0.0)

            if get_last_layer_activation(model) in ['Softmax', 'Sigmoid']:
                targets = model(training_observations)
                if get_balance(targets) < self.config["targets_perp_min_thresh"]:
                    if checks[-1]:
                        self.error_msg.append(self.main_msgs['last_bias'])
                    elif not checks[-1]:
                        bias_indices = torch.argsort(b_array)
                        probas_indices = torch.argsort(get_probas(targets))
                        if not torch.equal(bias_indices, probas_indices):
                            self.error_msg.append(self.main_msgs['ineff_bias_cls'])

            if not torch.tensor(checks).all():
                self.error_msg.append(self.main_msgs['zero_bias'])
