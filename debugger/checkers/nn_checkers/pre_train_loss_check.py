from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import get_loss, get_model_weights_and_biases
import torch


def get_config():
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {"Period": 0,
              "init_loss": {"size_growth_rate": 2, "size_growth_iters": 5, "dev_ratio": 1.0}}
    return config


class PreTrainLossCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="PreTrainLoss", config=get_config())

    def run(self, labels, predictions, loss_fn, model):
        """
        Run multiple checks on the loss function and its generated values. This checker runs before the training and
        does the following checks on the initial loss outputs :

        The checks include: 1.Ensure correct reduction of the loss function. this is more useful for custom loss
        function implementations. Loss is calculated with increasing batch sizes to confirm proportional increase,
        indicating that the loss estimation is based on average, not sum.

        2. Verify that the optimizer's initial loss is as expected. An initial loss approximation can be
        calculated for most of the predefined loss functions. This function compares the model's initial loss to the
        calculated approximation.

        Args:
            labels (Tensor): The ground truth of the initial observationsTargets used in the loss function (for
             example the labels in the DQN are the Q_target).

            predictions (Tensor): The outputs of the model in the initial set of observations. loss_fn (function):
            The loss function. model (nn.Module): The model to be trained.
        """
        if not self.check_period():
            return
        losses = []
        n = self.config["init_loss"]["size_growth_rate"]
        while n <= (self.config["init_loss"]["size_growth_rate"] * self.config["init_loss"]["size_growth_iters"]):
            derived_batch_y = torch.cat([labels] * n, dim=0)
            derived_predictions = torch.cat(n * [predictions.clone().detach().cpu()], dim=0)
            loss_value = float(get_loss(derived_predictions, derived_batch_y, loss_fn))
            losses.append(loss_value)
            n *= self.config["init_loss"]["size_growth_rate"]
        rounded_loss_rates = [round(losses[i + 1] / losses[i]) for i in range(len(losses) - 1)]
        equality_checks = sum(
            [(loss_rate == self.config["init_loss"]["size_growth_rate"]) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):
            self.error_msg.append(self.main_msgs['poor_reduction_loss'])

        initial_loss = float(get_loss(predictions, labels, loss_fn))
        initial_weights, _ = get_model_weights_and_biases(model)
        number_of_actions = list(initial_weights.items())[-1][1].shape[0]
        expected_loss = -torch.log(torch.tensor(1 / number_of_actions))
        err = torch.abs(initial_loss - expected_loss)
        # TODO: this function may only work on the cross entropy loss
        if err >= self.config["init_loss"]["dev_ratio"] * expected_loss:
            self.error_msg.append(self.main_msgs['poor_init_loss'].format(round((err / expected_loss), 3)))
