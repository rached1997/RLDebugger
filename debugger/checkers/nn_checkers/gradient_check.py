import torch
from debugger.debugger_interface import DebuggerInterface
from torch.autograd import gradcheck


# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.

def get_config():
    config = {"Period": 0,
              "sample_size": 3,
              "delta": 0.0001,
              "relative_err_max_thresh": 0.01}
    return config


class GradientCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="Gradient", config=get_config())

    def run(self, loss_fn):
        error_msgs = list()

        inputs = (torch.randn(self.config["sample_size"], dtype=torch.double, requires_grad=True),
                  torch.randn(self.config["sample_size"], dtype=torch.double, requires_grad=True))

        theoretical_numerical_check = gradcheck(loss_fn, inputs, eps=self.config["delta"],
                                                rtol=self.config["relative_err_max_thresh"])

        if not theoretical_numerical_check:
            error_msgs.append(self.main_msgs['grad_err'].format(self.config["relative_err_max_thresh"]))
        return error_msgs
