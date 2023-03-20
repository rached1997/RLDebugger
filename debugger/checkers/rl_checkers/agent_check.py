import copy
import random
import torch
from debugger.config_data_classes.rl_checkers.agent_config import AgentConfig
from debugger.debugger_interface import DebuggerInterface
import torch.nn.functional as F


class AgentCheck(DebuggerInterface):
    """
    This class performs checks on the neural networks composing the agent, ensuring they are being updated and
    interacting correctly. This class of checks is dedicated to Q-Learning-based and Hybrid RL algorithms.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
            * _old_target_model_params: a dictionary containing the last parameters of the target network before being
            updated.
            * _old_training_data: a fixed batch of data used to measure the KL divergence. This data can be seen as
            the benchmarking data.
            * _old_model_output: the main model's predictions on the _old_training_data before being updated. This is
            useful to measure the KL divergence.
        """
        super().__init__(check_type="Agent", config=AgentConfig)
        self._old_target_model_params = None
        self._old_training_data = None
        self._old_model_output = None

    def run(
        self,
        model,
        target_model,
        actions_probs,
        target_model_update_period,
        observations,
        target_net_update_fraction,
    ) -> None:
        """
        ------------------------------------   I. Introduction of the Agent Check  ------------------------------------

        This class checks whether the agent's neural networks are being updated and coordinated correctly. In many
        DRL algorithms the agent is composed of a main and target network, with the target being a copy of the main.
        The two networks are updated periodically at different periods to enhance the stability and efficiency of
        the learning process.
            * This class ensures that the target network is updated correctly when it reaches the update period ,
              and that it has a fixed parameters in the other steps. For the main network's parameters, it's essential to
              verify that it has parameters different from the target network's parameters, when it's not the target
              network's update period. In addition, it is also important to check that the main network is used to
              predict the next action and not hte target network, as using the target network can cause an unstable
              learning process.
            * Additionally, this class checks that the main network's updates are smooth and do not result in
            catastrophic forgetting, which can lead to an unstable and unreliable learning process. To do this we
            measure the KL-divergence of the predictions of the model over successive updates of the model to see
            whether it's predictions are diverging or not

        Note that the update of the target model can be achieved through one of two forms: hard update or soft update.
            * In a hard update, the weights of the target network are directly copied from the main network.
            * In a soft update, the weights of the target network are gradually updated over time by interpolating
              between the weights of the target network and the main network.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        This class is responsible for performing several checks on the agent's behavior during the learning
            process. It does the following checks :
            (1) Verifies that the target model has been updated at the correct period with the correct parameter values.
            (2) Verifies that the main and target model have different parameter values when it's not the target
                network's update period.
            (3) Verifies that the agent's predictions are being made by the correct model.
            (4) Detects significant divergence in the model's behavior between successive updates, which can indicate
                learning instability.

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are
            - Incorrect network updates:
                * Target network is not updated at the correct period (checks triggered: 1, 2)
                * Target network is not updated with the values of the main network (whether it's a hard or soft update)
                  (checks triggered: 1)
            - Confusion between the main and target networks (checks triggered: 1, 2, 3)
            - Unstable learning process (checks triggered: 4)
            - Coding errors (checks triggered: 1, 2, 3)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues :
            - Check whether the target model is being updated at the correct time (checks that can be fixed: 1,2,3).
            - Check if you haven't mixed the main and target network:
                - You are using the main model to predict the next action (checks that can be fixed: 3)
                - You are updating the target network with the parameters of the main network (checks that can be fixed: 3)
            - Make sure the main model is updated correctly (checks that can be fixed: 4).
            - Check the models' hyperparameters to ensure that they are appropriately set (checks that can be fixed: 4).
            - Consider changing the update period of the target network to address learning instability issues (checks
            that can be fixed: 4).

        Examples
        --------
        To perform agent checks, the debugger needs to be called when updating the main and target networks. Note that
        the debugger needs to be called before performing the backward prop (.backward()) and the update of target
        network (.update_target()).

        >>> from debugger import rl_debugger
        >>> ...
        >>> batch = replay_buffer.sample(batch_size=32)
        >>> rl_debugger.debug(model=qnet, target_model=target_qnet, target_model_update_period=period,
        >>>                   target_net_update_fraction=update_fraction, training_observations=batch["state"])
        >>> loss = loss_fn(pred_qvals, q_targets).mean()
        >>> loss.backward()
        >>> update_target()

        Args:
            model (nn.Module): the main model
            target_model (nn.Module): the target model
            actions_probs (Tensor): the predictions on the observations
            target_model_update_period (float): The period of the target network update, (given as the number of steps taken).
            training_observations (Tensor): the batch of observations collected during the training used to obtain
            actions_probs
            target_net_update_fraction (float): The fraction of the target model parameters to be updated in each
            target model update (usefull for the soft upadte). Note if you are using the hard update set its value to 1
        """
        if self._old_target_model_params is None:
            self._old_target_model_params = copy.deepcopy(list(target_model.modules()))
        self.save_observation_to_buffer(observations)
        self.check_main_target_models_behaviour(
            target_model,
            model,
            target_net_update_fraction,
            target_model_update_period,
        )

        if self.check_period():
            self.check_wrong_model_output(model, observations, actions_probs)
            if len(self._old_training_data) >= self.config.start:
                self.check_kl_divergence(model)
        if len(self._old_training_data) >= self.config.start:
            self._old_model_output = model(self._old_training_data)

    def check_main_target_models_behaviour(
        self,
        target_model,
        model,
        target_net_update_fraction,
        target_model_update_period,
    ):
        """
        Checks whether the main and target models are being updated correctly during the learning process. This function
        verifies that the target model has been updated correctly when reaching the update period. It also checks
        whether both the main and target networks have different parameters during training.

        Args:
            target_model (nn.Module): the weights and biases of the target model.
            model (nn.Module): the weights and biases of the main model.
            target_net_update_fraction (float): The fraction of the target model parameters to be updated in each target model update.
            target_model_update_period (float):  The period of the target network update, (given as the number of steps taken).

        """
        if (
            (((self.step_num - 1) % target_model_update_period) == 0)
            and (self.step_num > 1)
            and not self.config.target_update.disabled
        ):
            all_equal, _, _ = self.compare_random_layers(
                model, target_model, target_net_update_fraction
            )
            if not all_equal:
                self.error_msg.append(self.main_msgs["target_network_not_updated"])
            self._old_target_model_params = copy.deepcopy(list(target_model.modules()))

        elif (
            (((self.step_num - 2) % target_model_update_period) == 0)
            and (self.step_num > 2)
            and (not self.config.similarity.disabled)
        ):
            (
                all_equal,
                layer_idx,
                random_target_model_layer,
            ) = self.compare_random_layers(
                model, target_model, target_net_update_fraction
            )
            if all_equal:
                self.error_msg.append(self.main_msgs["similar_target_and_main_network"])
            if not torch.equal(
                self._old_target_model_params[layer_idx].weight,
                random_target_model_layer,
            ):
                self.error_msg.append(self.main_msgs["target_network_changing"])

    def check_wrong_model_output(self, model, observations, action_probs):
        """
        Checks whether the wrong model is being used to predict the following action during the learning process.

        Args:
            model (nn.Module): the main model
            observations (Tensor): a batch of observations collected during the training
            action_probs (Tensor): the predictions on the observations
        """
        if self.config.wrong_model_out.disabled:
            return
        pred_qvals = model(
            torch.tensor(observations, device=self.device).unsqueeze(dim=0)
        )
        if not torch.equal(action_probs, pred_qvals):
            self.error_msg.append(self.main_msgs["using_the_wrong_network"])

    def check_kl_divergence(self, model):
        """
        checks if there is a normal divergence in the model's predictions before and after being updated. A normal KL
        divergence value should be small but positive, otherwise there is a drastic change in the model's behaviour

        Args:
            model (nn.Module): the main model

        Returns:

        """
        if self.iter_num > 1 and (not self.config.kl_div.disabled):
            new_model_output = model(self._old_training_data)
            if not torch.allclose(
                torch.sum(new_model_output, dim=1),
                torch.ones(new_model_output.shape[0], device=self.device),
            ):
                new_model_output = F.softmax(new_model_output, dim=1)
                self._old_model_output = F.softmax(self._old_model_output, dim=1)
            kl_div = F.kl_div(
                torch.log(new_model_output),
                self._old_model_output,
                reduction="batchmean",
            )
            if torch.any(kl_div > self.config.kl_div.div_threshold):
                self.error_msg.append(
                    self.main_msgs["kl_div_high"].format(
                        kl_div, self.config.kl_div.div_threshold
                    )
                )

    def compare_random_layers(self, model, target_model, target_net_update_fraction):
        """
        Compares the weights values of a random layer in the main and target network (the layer shouldn't be neither
        the input nor the output layer)

        Args:
            model (nn.Module): the main model
            target_model (nn.Module): the target model
            target_net_update_fraction (float): The fraction of the target model parameters to be updated in each

        return (boolean): True if the random layer has the same weights in both main and target networks, otherwise
        false
        """
        layers = [
            i
            for i, module in enumerate(model.modules())
            if (hasattr(module, "bias") or hasattr(module, "weight"))
        ][1:-1]
        layer_idx = random.choice(layers)

        random_main_model_layer = list(model.modules())[layer_idx].weight
        random_target_model_layer = list(target_model.modules())[layer_idx].weight

        all_equal = torch.equal(
            random_target_model_layer,
            (1 - target_net_update_fraction)
            * self._old_target_model_params[layer_idx].weight
            + target_net_update_fraction * random_main_model_layer,
        )

        return all_equal, layer_idx, random_target_model_layer

    def save_observation_to_buffer(self, observations):
        """
        Save the observations to the buffer self._old_training_data

        args:
            observations (Tensor): The tensor of the observation to be saved
        """
        reshaped_observation = torch.tensor(observations, device=self.device).unsqueeze(
            dim=0
        )
        if self._old_training_data is None:
            self._old_training_data = reshaped_observation
        elif len(self._old_training_data) < self.config.start:
            self._old_training_data = torch.cat(
                [self._old_training_data, reshaped_observation], dim=0
            )
