import copy
import numpy as np
import torch.nn

from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import smoothness
from debugger.utils.utils import are_significantly_different
from debugger.utils.model_params_getters import get_loss


def get_config():
    config = {
        "Period": 0,
        "single_batch_size": 16,
        "total_iters": 100,
        "abs_loss_min_thresh": 1e-8,
        "loss_min_thresh": 0.00001,
        "smoothness_max_thresh": 0.95,
        "mislabeled_rate_max_thresh": 0.05,
        "mean_error_max_thresh": 0.001,
        "sample_size_of_losses": 100,
        "Instance_wise_Operation": {"sample_size": 32, "trials": 10}}

    return config


class PreTrainProperFittingCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="PreTrainProperFitting", config=get_config())

    def run(self, observations, labels, actions, opt, model, loss_fn):
        if not self.check_period():
            return

        real_losses = self.overfit_verification(model, opt, observations, labels, actions, loss_fn)
        if not real_losses:
            return
        if not self.regularization_verification(real_losses):
            return

        fake_losses = self.input_dependency_verification(model, opt, observations, labels, actions, loss_fn)
        if not fake_losses:
            return

        stability_test = np.array([self._loss_is_stable(loss_value) for loss_value in (real_losses + fake_losses)])
        if (stability_test == False).any():
            last_real_losses = real_losses[-self.config['sample_size_of_losses']:]
            last_fake_losses = fake_losses[-self.config['sample_size_of_losses']:]
            if not (are_significantly_different(last_real_losses, last_fake_losses)):
                self.error_msg.append(self.main_msgs['data_dep'])

    def _loss_is_stable(self, loss_value):
        if np.isnan(loss_value):
            self.error_msg.append(self.main_msgs['nan_loss'])
            return False
        if np.isinf(loss_value):
            self.error_msg.append(self.main_msgs['inf_loss'])
            return False
        return True

    def input_dependency_verification(self, model, opt, derived_batch_x, derived_batch_y, actions, loss_fn):
        zeroed_model = copy.deepcopy(model)
        zeroed_opt = opt.__class__(zeroed_model.parameters(), )

        zeroed_batch_x = torch.zeros_like(derived_batch_x)
        fake_losses = []
        zeroed_model.train(True)
        for i in range(self.config["total_iters"]):
            zeroed_opt.zero_grad()
            outputs = zeroed_model(zeroed_batch_x)
            outputs = outputs[torch.arange(outputs.size(0)), actions]
            fake_loss = float(get_loss(outputs, derived_batch_y, loss_fn))
            fake_losses.append(fake_loss)
            if not (self._loss_is_stable(fake_loss)):
                return False
        return fake_losses

    def overfit_verification(self, model, opt, derived_batch_x, derived_batch_y, actions, loss_fn):
        overfit_opt = opt.__class__(model.parameters(), )

        real_losses = []
        model.train(True)
        for i in range(self.config["total_iters"]):
            overfit_opt.zero_grad()
            outputs = model(derived_batch_x)
            outputs = outputs[torch.arange(outputs.size(0)), actions]
            loss_value = get_loss(outputs, derived_batch_y, loss_fn)
            loss_value.backward()
            overfit_opt.step()
            real_losses.append(loss_value.item())
            if not (self._loss_is_stable(loss_value.item())):
                self.error_msg.append(self.main_msgs['underfitting_single_batch'])
                return False
        return real_losses

    def regularization_verification(self, real_losses):
        loss_smoothness = smoothness(np.array(real_losses))
        min_loss = np.min(np.array(real_losses))
        if min_loss <= self.config['abs_loss_min_thresh'] or (
                min_loss <= self.config['loss_min_thresh'] and loss_smoothness > self.config['smoothness_max_thresh']):
            self.error_msg.append(self.main_msgs['zero_loss'])
            return False
        return True


