import torch
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_model_weights_and_biases
from debugger.utils.utils import get_probas, get_balance


def get_config():
    config = {"Period": 0,
              "labels_perp_min_thresh": 0.5
              }
    return config


class BiasCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="Bias", config=get_config())

    def run(self, model, observations):
        error_msg = list()
        _, initial_biases = get_model_weights_and_biases(model)
        if not initial_biases:
            error_msg.append(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(torch.sum(b_array) == 0.0)

            targets = model(torch.tensor(observations))
            if get_balance(targets) < self.config["labels_perp_min_thresh"]:
                if checks[-1]:
                    error_msg.append(self.main_msgs['last_bias'])
                elif not checks[-1]:
                    bias_indices = torch.argsort(b_array)
                    probas_indices = torch.argsort(get_probas(targets))
                    if not torch.equal(bias_indices, probas_indices):
                        error_msg.append(self.main_msgs['ineff_bias_cls'])

            if not torch.tensor(checks).all():
                error_msg.append(self.main_msgs['zero_bias'])
        return error_msg


