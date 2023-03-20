import torch
from debugger.config_data_classes.nn_checkers.gradient_config import GradientConfig
from debugger.debugger_interface import DebuggerInterface
from torch.autograd import gradcheck


class GradientCheck(DebuggerInterface):
    """
    The check is in charge of verifying the bias values during training.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        super().__init__(check_type="Gradient", config=GradientConfig)

    def run(self, loss_fn):
        """
        -----------------------------------   I. Introduction of the Gradient Check  -----------------------------------

        This class is responsible for performing checks on the gradients of the model. Gradients represent the
        direction towards which the weights will be updated, making it a crucial feature for updating the parameters
        of the model during training. It is important to ensure that the gradient is behaving correctly and being
        calculated accurately, as any errors can lead to suboptimal model performance. This class helps identify any
        issues with the gradients and ensures that they are being calculated correctly.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        The Gradient check class performs the following pre-check:
            (1) compares the numerical gradient with the analytic gradient. To see if the gradient is being
            calculated correctly

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are
            - Wrong calculation of the gradient (checks triggered : 1)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues:
            - Verify that the gradient is calculated correctly ( checks tha can be fixed: 1)

        Examples
        --------

        To perform gradient checks, the debugger needs to be called after defining the loss function. The ideal location
        would be in the first call of ".debug()".

        >>> from debugger import rl_debugger
        >>> ...
        >>> loss_function = torch.nn.SmoothL1Loss
        >>> ...
        >>> env = gym.make("CartPole-v1")
        >>> rl_debugger.debug(environment=env, loss_fn=loss_function)

        Please note that the environment needs to be always passed in the first call of ".debug()". The environment is
        the only parameter required for the debugger to properly operate.

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
