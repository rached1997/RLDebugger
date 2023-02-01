from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
import torch


def get_config():
    config = {"Period": 0,
              "init_loss": {"size_growth_rate": 2, "size_growth_iters": 5, "dev_ratio": 1.0}}
    return config


class PreTrainLossCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="PreTrainLoss", config=get_config())

    def run(self, labels, predictions, loss, model):
        if not self.check_period():
            return
        losses = []
        n = self.config["init_loss"]["size_growth_rate"]
        while n <= (self.config["init_loss"]["size_growth_rate"] * self.config["init_loss"]["size_growth_iters"]):
            derived_batch_y = torch.cat([labels] * n, dim=0)
            derived_predictions = torch.cat(n * [predictions.clone().detach().cpu()], dim=0)
            loss_value = float(get_loss(derived_predictions, derived_batch_y, loss))
            losses.append(loss_value)
            n *= self.config["init_loss"]["size_growth_rate"]
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config["init_loss"]["size_growth_rate"]) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            self.error_msg.append(self.main_msgs['poor_reduction_loss'])

        initial_loss = float(get_loss(predictions, labels, loss))
        initial_weights, _ = get_model_weights_and_biases(model)
        number_of_actions = list(initial_weights.items())[-1][1].shape[0]
        expected_loss = -torch.log(torch.tensor(1 / number_of_actions))
        err = torch.abs(initial_loss - expected_loss)
        if err >= self.config["init_loss"]["dev_ratio"] * expected_loss:
            self.error_msg.append(self.main_msgs['poor_init_loss'].format(round((err / expected_loss), 3)))
