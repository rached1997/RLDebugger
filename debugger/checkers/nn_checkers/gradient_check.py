import torch
from debugger.debugger_interface import DebuggerInterface
from torch.autograd import gradcheck


def get_config():
    """
        Return the configuration dictionary needed to run the checkers.

        Returns:
            config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {"period": 1,
              "sample_size": 3,
              "delta": 0.0001,
              "relative_err_max_thresh": 0.01}
    return config


class GradientCheck(DebuggerInterface):
    """
    The check is in charge of verifying the gradient values during pre-training.
    """

    def __init__(self):
        super().__init__(check_type="Gradient", config=get_config())

    def run(self, loss_fn):
        """
        This function compares the numerical gradient with the analytic gradient. we perform a numerical gradient
        checking that consists of comparing the analytic and numerical calculated gradients (i.e., the gradient
        produced by the analytic formula and the centered finite difference approximation). Both gradients should be
        approximately equal for the same data points. This check is very useful when DL developers add hand-crafted
        math operations and gradient estimators. More details on theoretical proof of this function can be found
        here:
            -   https://link.springer.com/article/10.1007/s10710-017-9314-z
            -   https://cs231n.github.io/neural-networks-3/

        Args:
            loss_fn: (torch.nn.Module) the loss function of the model.

        Returns:

        """
        if not self.check_period():
            return

        inputs = (torch.randn(self.config["sample_size"], dtype=torch.double, requires_grad=True),
                  torch.randn(self.config["sample_size"], dtype=torch.double, requires_grad=True))

        # torch.autograd.gradcheck takes a tuple of tensors as input, check if your gradient evaluated with these
        # tensors are close enough to numerical approximations and returns True if they all verify this condition.
        theoretical_numerical_check = gradcheck(loss_fn, inputs, eps=self.config["delta"],
                                                rtol=self.config["relative_err_max_thresh"])

        if not theoretical_numerical_check:
            self.error_msg.append(self.main_msgs['grad_err'].format(self.config["relative_err_max_thresh"]))
