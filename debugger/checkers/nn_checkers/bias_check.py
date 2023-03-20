from debugger.config_data_classes.nn_checkers.bias_config import BiasConfig
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.model_params_getters import (
    get_model_weights_and_biases,
    get_last_layer_activation,
)
import torch

from debugger.utils.utils import get_balance, get_probas


class BiasCheck(DebuggerInterface):
    """
    The check is in charge of verifying the bias values during training.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
            * b_reductions: a dictionary tracking the bias of each layer
        """
        super().__init__(check_type="Bias", config=BiasConfig)
        self._b_reductions = dict()

    def run(self, model: torch.nn.Module, observations: torch.Tensor = None) -> None:
        """
        -----------------------------------   I. Introduction of the Bias Check  -----------------------------------

        The BiasCheck class is responsible for performing various checks on the neural network's bias. Bias is an
        essential feature in any neural network that helps it learn more complex and non-linear patterns in the data.
        It is crucial for reducing the underfitting of the neural network. Generally, bias should be initialized to
        values different from zero and during training, it should remain stable, i.e., it should not diverge,
        contain infinite or NaN values, or exhibit any other abnormal behavior. By ensuring that the bias is behaving
        normally, this class helps improve the overall performance and learning efficiency of the neural network.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        This class performs the following checks on the bias during the training:
            (1) Run the pre-checks on the bias values, before starting the
                a. Verifies the existence of the bias
                b. Checks if the bias of the last layer is non-zero.
                c. Validates if the bias of the last layer matches the label ratio.
                d. Confirms that the bias is not set to zero.
            (2) Validates the numerical stability of bias values during training (i.e. no inf or nan values)
            (3) Check if the bias values are diverging

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are
            - Bad initialisation of te bias values (checks triggered : 1)
            - Unstable learning process (checks triggered : 2,3)
            - Bad data preprocessing (checks triggered : 3)
            - The data training data (e.g. unbalances, doesn't have a signal) (checks triggered : 3)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues:
            - Reinitialize the biases ( checks that can be fixed: 1)
            - Stabilize the learning (e.g. change the hyperparameters, change the neural network's architecture) (
            checks tha can be fixed: 2,3)
            - Improve the data preprocessing towards reducing the noise contained in the data (checks that can be fixed: 3)

        Examples
        --------
        To perform bias checks, the debugger can be called when the RL agent is predictiong the action ("act()"
        function) or when the RL is updating its networks.
        The debugger needs "model" parameter only to perform these checks. The 'observations' parameter is
        automatically observed by debugger, and you don't need to pass it to the 'debug()' function.

        >>> from debugger import rl_debugger
        >>> ...
        >>> action, action_logprob, state_val, action_probs = policy_old.act(state)
        >>> rl_debugger.debug(model=policy_old.actor)

        In the context of DQN, the act() method is one location to invoke the debugger to perform bias checks.

        >>> from debugger import rl_debugger
        >>> ...
        >>> state, reward, done, _ = env.step(action)
        >>> qvals = qnet(state)
        >>> rl_debugger.debug(model=qnet)

         The debugger can also be called in the update() method to perform bias checks.

        >>> from debugger import rl_debugger
        >>> ...
        >>> batch = replay_buffer.sample(batch_size=32)
        >>> pred_qvals = qnet(batch['state'])
        >>> rl_debugger.debug(model=qnet)
        >>> loss = loss_fn(pred_qvals, q_targets).mean()
        >>> loss.backward()

        If you feel that this check is slowing your code, you can increase the value of "skip_run_threshold" in
        BiasConfig.

        Args:
            model (nn.model): The model being trained
            observations (Tensor): Initial sample of observations.

        Returns:
            None
        """
        if self.skip_run(self.config.skip_run_threshold):
            return
        if self.iter_num == 1:
            self.run_pre_checks(model, observations)
        for name, param in model.named_parameters():
            if "bias" in name:
                b_name = name.split(".bias")[0]
                b_array = param.data
                b_reductions = self.update_b_reductions(b_name, b_array)
                if (self.iter_num >= self.config.start) and self.check_period():
                    if self.check_numerical_instabilities(b_name, b_array):
                        continue
                    self.check_divergence(b_name, b_reductions)

    def update_b_reductions(
        self, bias_name: str, bias_array: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates and save the biases periodically. At each step, the mean value of the biases is stored.

        Args:
            bias_name (str): The name of the layer to be validated.
            bias_array (Tensor): The biases obtained from the specified layer.

        Returns:
            (Tensor): all average biases obtained during training.
        """
        if bias_name not in self._b_reductions:
            self._b_reductions[bias_name] = torch.tensor([], device=self.device)
        self._b_reductions[bias_name] = torch.cat(
            (self._b_reductions[bias_name], torch.mean(torch.abs(bias_array)).view(1))
        )
        return self._b_reductions[bias_name]

    def check_numerical_instabilities(
        self, bias_name: str, bias_array: torch.Tensor
    ) -> bool:
        """
        Validates the numerical stability of bias values during training.

        Args:
            bias_name: (str) The name of the layer to be validated.
            bias_array: (Tensor): The biases obtained from the specified layer.

        Returns:
            (bool): True if there is any NaN or infinite value present, False otherwise.
        """
        if self.config.numeric_ins.disabled:
            return False
        if torch.isinf(bias_array).any():
            self.error_msg.append(self.main_msgs["b_inf"].format(bias_name))
            return True
        if torch.isnan(bias_array).any():
            self.error_msg.append(self.main_msgs["b_nan"].format(bias_name))
            return True
        return False

    def check_divergence(self, bias_name: str, bias_reductions: torch.Tensor) -> None:
        """
        This function check bias divergence, as biases risk divergence, and may go towards inf. Biases can become
        huge in cases when features (observation) do not adequately explain the predicted outcome or are ineffective.
        This function automates a verification routine that watches continuously the absolute averages of bias
         are not diverging. More details on theoretical proof of this function can be found here:
            - https://arxiv.org/pdf/2204.00694.pdf

        Args:
            bias_name: (str) The name of the layer to be validated.
            bias_reductions:

        Returns:

        """
        if self.config.div.disabled:
            return
        if bias_reductions[-1] > self.config.div.mav_max_thresh:
            self.error_msg.append(
                self.main_msgs["b_div_1"].format(
                    bias_name, bias_reductions[-1], self.config.div.mav_max_thresh
                )
            )
        elif len(bias_reductions) >= self.config.div.window_size:
            inc_rates = (
                bias_reductions[1 - self.config.div.window_size :]
                / bias_reductions[-self.config.div.window_size : -1]
            )
            if (inc_rates >= self.config.div.inc_rate_max_thresh).all():
                self.error_msg.append(
                    self.main_msgs["b_div_2"].format(
                        bias_name,
                        max(inc_rates),
                        self.config.div.inc_rate_max_thresh,
                    )
                )

    def run_pre_checks(self, model, observations) -> None:
        """
        This function performs multiple checks on the bias initial values of the model:

        (1) Verifies the existence of the bias
        (2) Checks if the bias of the last layer is non-zero when the model's output in the initial observation set
        is imbalanced.
        (3) Validates if the bias of the last layer matches the label ratio when the output of the model in the
        initial observation set is imbalanced, using the formula bi = log(pi / (1-pi)), where pi is the proportion of
        observations of the label (actions) corresponding to the bias bi of unit i.
        (4) Confirms that the bias is not set to zero.

        Args:
        model (torch.nn.Module): The model that is being trained.
        observations (torch.Tensor): A sample of observations collected before the start of the training process.
        """
        _, initial_biases = get_model_weights_and_biases(model)
        if not initial_biases:
            self.error_msg.append(self.main_msgs["need_bias"])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(torch.sum(b_array) == 0.0)

            if get_last_layer_activation(model) in ["Softmax", "Sigmoid"]:
                targets = model(
                    torch.tensor(observations, device=self.device).unsqueeze(dim=0)
                )
                if get_balance(targets) < self.config.targets_perp_min_thresh:
                    if checks[-1]:
                        self.error_msg.append(self.main_msgs["last_bias"])
                    elif not checks[-1]:
                        bias_indices = torch.argsort(b_array)
                        probas_indices = torch.argsort(get_probas(targets))
                        if not torch.equal(bias_indices, probas_indices):
                            self.error_msg.append(self.main_msgs["ineff_bias_cls"])

            if not torch.tensor(checks).all():
                self.error_msg.append(self.main_msgs["zero_bias"])
