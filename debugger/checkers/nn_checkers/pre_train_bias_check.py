import torch
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_model_weights_and_biases, get_last_layer_activation
from debugger.utils.utils import get_probas, get_balance


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """

    config = {"Period": 0,
              "labels_perp_min_thresh": 0.5
              }
    return config


class PreTrainBiasCheck(DebuggerInterface):
    """
    The check is in charge of verifying the bias values during pre-training.
    """

    def __init__(self):
        super().__init__(check_type="PreTrainBias", config=get_config())

    def run(self, model: torch.nn.Module, observations: torch.Tensor) -> None:
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
        if not self.check_period():
            return
        _, initial_biases = get_model_weights_and_biases(model)
        if not initial_biases:
            self.error_msg.append(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(torch.sum(b_array) == 0.0)

            if get_last_layer_activation(model) in ['Softmax', 'Sigmoid']:
                targets = model(observations)
                if get_balance(targets) < self.config["labels_perp_min_thresh"]:
                    if checks[-1]:
                        self.error_msg.append(self.main_msgs['last_bias'])
                    elif not checks[-1]:
                        bias_indices = torch.argsort(b_array)
                        probas_indices = torch.argsort(get_probas(targets))
                        if not torch.equal(bias_indices, probas_indices):
                            self.error_msg.append(self.main_msgs['ineff_bias_cls'])

            if not torch.tensor(checks).all():
                self.error_msg.append(self.main_msgs['zero_bias'])


