import copy
import numpy as np
import torch.nn
from debugger.config_data_classes.nn_checkers.proper_fitting_config import (
    ProperFittingConfig,
)
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import smoothness
from debugger.utils.utils import are_significantly_different
from debugger.utils.model_params_getters import get_loss


class ProperFittingCheck(DebuggerInterface):
    """
    The check in charge of verifying the proper fitting of the DNN before training.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        super().__init__(check_type="ProperFitting", config=ProperFittingConfig)

    def run(self, training_observations, targets, actions, opt, model, loss_fn):
        """
        --------------------------------- I. Introduction of the Proper Fitting Check ---------------------------------

        The ProperFittingCheck function evaluates the convergence ability  of the neural network model by testing its
        ability to fit a small sample of data.

        The function trains the model on a sample of data and then uses it to predict the output values for the same
        sample. It then compares the predicted values with the actual values to calculate the model's accuracy.

        ------------------------------------------   II. The performed checks  -----------------------------------------
        The ProperFittingCheck function performs the following checks:
            (1) Checks the stability of the loss function
            (2) Checks if there is a Regularization
            (3) Input dependency verification, which compares training performance on real observations and zeroed
                observations

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are
            - Bad architecture of the neural network (checks triggered : 1,2,3)
            - Bad hyperparameters (checks triggered : 1,2,3)
            - Bad initialization of the model's parameters (checks triggered : 1,2,3)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues:
            - Change the architecture of the neural network ( checks tha can be fixed: 1,2,3)
            - Do a hyperparameter tuning ( checks tha can be fixed: 1,2,3)
            - Reinitialize the model ( checks tha can be fixed: 1,2,3)



        Examples
        --------
        To perform value function checks, the debugger needs to be called when updating the agent.

        >>> from debugger import rl_debugger
        >>> ...
        >>> next_qvals = target_qnet(next_states)
        >>> next_qvals, _ = torch.max(next_qvals, dim=1)
        >>> batch = replay_buffer.sample(batch_size=32)
        >>> q_targets = batch["reward"] + discount_rate * next_qvals * (1 - batch["done"])
        >>> rl_debugger.debug(training_observations=batch["state"], targets=q_targets.detach(), actions=actions,
        >>>                   opt=optimizer, model=qnet, loss_fn=loss_fn)
        >>> loss = loss_fn(pred_qvals, q_targets).mean()

        Args:
            training_observations (Tensor): Initial sample of observations.
            targets (Tensor): Ground truth of the initial observations.
            actions (Tensor): Predicted actions for the initial set of observations.
            opt (function): Optimizer function.
            model (nn.Module): Model to be trained.
            loss_fn (function): Loss function.
        """
        if self.config.disabled:
            return
        model = copy.deepcopy(model)
        opt = copy.deepcopy(opt)
        if not self.check_period():
            return

        real_losses = self.overfit_verification(
            model, opt, training_observations, targets, actions, loss_fn
        )
        if not real_losses:
            return

        if not self.regularization_verification(real_losses):
            return

        fake_losses = self.input_dependency_verification(
            model, opt, training_observations, targets, actions, loss_fn
        )
        if not fake_losses:
            return

        stability_test = np.array(
            [
                self._loss_is_stable(loss_value)
                for loss_value in (real_losses + fake_losses)
            ]
        )
        if (stability_test == False).any():
            last_real_losses = real_losses[-self.config.sample_size_of_losses :]
            last_fake_losses = fake_losses[-self.config.sample_size_of_losses :]
            if not (are_significantly_different(last_real_losses, last_fake_losses)):
                self.error_msg.append(self.main_msgs["data_dep"])

    def _loss_is_stable(self, loss_value):
        """
        Evaluate the stability of the loss function by checking for the presence of NaN or infinite values.

        Args:
            loss_value (float): The value of the loss function as calculated during training on the initial set of observations.

        Returns:
            bool: True if the loss value is finite (not NaN or infinite), False otherwise.
        """

        if np.isnan(loss_value):
            self.error_msg.append(self.main_msgs["nan_loss"])
            return False
        if np.isinf(loss_value):
            self.error_msg.append(self.main_msgs["inf_loss"])
            return False
        return True

    def input_dependency_verification(
        self, model, opt, derived_batch_x, derived_batch_y, actions, loss_fn
    ):
        """
        This function evaluates the stability of the loss generated by training the model on a batch of zeroed data.

        Args:
            model (nn.Module): The model to be trained.
            opt (function): The optimization function used to update the model's parameters.
            derived_batch_x (Tensor): The input observations used for training.
            derived_batch_y (Tensor): The ground truth targets for the input observations.
            actions (Tensor): The predicted actions for the initial set of observations.
            loss_fn (function): The loss function used to evaluate the model's performance.

        Returns:
            False if the generated loss is unstable, otherwise a list of the collected losses (fake_losses).
        """

        zeroed_model = copy.deepcopy(model)
        zeroed_opt = opt.__class__(
            zeroed_model.parameters(),
        )

        zeroed_batch_x = torch.zeros_like(derived_batch_x)
        fake_losses = []
        zeroed_model.train(True)
        for i in range(self.config.total_iters):
            zeroed_opt.zero_grad()
            outputs = zeroed_model(zeroed_batch_x)
            outputs = outputs[torch.arange(outputs.size(0)), actions]
            fake_loss = float(get_loss(outputs, derived_batch_y, loss_fn))
            fake_losses.append(fake_loss)
            if not (self._loss_is_stable(fake_loss)):
                return False
        return fake_losses

    def overfit_verification(
        self, model, opt, derived_batch_x, derived_batch_y, actions, loss_fn
    ):
        """
        This function tracks the losses during training the model on the initial sample of observations.

        Args:
            model (nn.Module): The model to be trained.
            opt (function): The optimizer function.
            derived_batch_x (Tensor): The initial sample of observations, representing the input to the `run` function.
            derived_batch_y (Tensor): The ground truth of the initial observations, used in the loss function.
            actions (Tensor): The predicted actions for the initial set of observations.
            loss_fn (function): The loss function to be used.

        Returns:
            False if the loss collected are not stable (check the function _loss_is_stable for more details), means the
            model is unable to fit a single batch of observations properly. Otherwise, it returns real_losses, a list of
             the losses collected during training on the initial sample of observations.
        """
        overfit_opt = opt.__class__(
            model.parameters(),
        )

        real_losses = []
        model.train(True)
        for i in range(self.config.total_iters):
            overfit_opt.zero_grad()
            outputs = model(derived_batch_x)
            outputs = outputs[torch.arange(outputs.size(0)), actions]
            loss_value = get_loss(outputs, derived_batch_y, loss_fn)
            loss_value.backward()
            overfit_opt.step()
            real_losses.append(loss_value.item())
            if not (self._loss_is_stable(loss_value.item())):
                self.error_msg.append(self.main_msgs["underfitting_single_batch"])
                return False
        return real_losses

    def regularization_verification(self, real_losses):
        """
        This functions checks if there is a regularization or not. Thus, it checks if the losses obtained through
        training the model on the sample of observations are inferior to a predefined threshold (check the parameter
        abs_loss_min_thresh in the gget_config function's output) or that the smoothness of the loss is superior to a
        predefined threshold (check smoothness_max_thresh in the function get_config's output )

        Args:
            real_losses (list): losses obtained through training the model on the sample of observations

        Returns:
            (boolean) False if there is no regularization, True otherwise

        """
        loss_smoothness = smoothness(np.array(real_losses))
        min_loss = np.min(np.array(real_losses))
        if min_loss <= self.config.abs_loss_min_thresh or (
            min_loss <= self.config.loss_min_thresh
            and loss_smoothness > self.config.smoothness_max_thresh
        ):
            self.error_msg.append(self.main_msgs["zero_loss"])
            return False
        return True
