import torch
from debugger.debugger_interface import DebuggerInterface
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.


class GradientCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Gradient"
        self.check_period = check_period

    def run(self, predictions, labels, loss):
        error_msgs = list()

        inputs = (torch.randn(self.config["sample_size"], dtype=torch.double, requires_grad=True),
                  torch.randn(self.config["sample_size"], dtype=torch.double, requires_grad=True))

        theoretical_numerical_check = gradcheck(loss, inputs, eps=self.config["delta"],
                                                rtol=self.config["relative_err_max_thresh"])

        if not theoretical_numerical_check:
            error_msgs.append(self.main_msgs['grad_err'].format(self.config["relative_err_max_thresh"]))
        return error_msgs
