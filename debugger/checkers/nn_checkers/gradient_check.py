import torch
from debugger.config_data_classes.nn_checkers.gradient_config import GradientConfig
from debugger.debugger_interface import DebuggerInterface
from torch.autograd import gradcheck


class GradientCheck(DebuggerInterface):
    """
    The check is in charge of verifying the gradient values during pre-training.
    """

    def __init__(self):
        super().__init__(check_type="Gradient", config=GradientConfig)

    def run(self, loss_fn):
        """
        The gradient represents the directions towards which the weights will be updated. Its the feature responsible
        for updating the parameters of the model. Thus it's crucial to make sure that the gradient is doing the right
        behaviour and check if it's calculated correctly

        This Gradient check performs the following pre-check:
            (1) compares the numerical gradient with the analytic gradient. we perform a numerical gradient
            checking that consists of comparing the analytic and numerical calculated gradients (i.e., the gradient
            produced by the analytic formula and the centered finite difference approximation). Both gradients should be
            approximately equal for the same data points. This check is very useful when DL developers add hand-crafted
            math operations and gradient estimators. More details on theoretical proof of this function can be found
            here: - https://link.springer.com/article/10.1007/s10710-017-9314-z -
            https://cs231n.github.io/neural-networks-3/

        The potential root causes behind the warnings that can be detected are
            - Wring calculation of the gradient (checks triggered : 1)

        The recommended fixes for the detected issues:
            - Verify that the gradient is calculated correctly ( checks tha can be fixed: 1)

        Args:
            loss_fn: (torch.nn.Module) the loss function of the model.

        Returns:

        """
        if not self.check_period():
            return

        inputs = (
            torch.randn(
                self.config.sample_size, dtype=torch.double, requires_grad=True
            ),
            torch.randn(
                self.config.sample_size, dtype=torch.double, requires_grad=True
            ),
        )

        # torch.autograd.gradcheck takes a tuple of tensors as input, check if your gradient evaluated with these
        # tensors are close enough to numerical approximations and returns True if they all verify this condition.
        theoretical_numerical_check = gradcheck(
            loss_fn,
            inputs,
            eps=self.config.delta,
            rtol=self.config.relative_err_max_thresh,
        )

        if not theoretical_numerical_check:
            self.error_msg.append(
                self.main_msgs["grad_err"].format(self.config.relative_err_max_thresh)
            )
