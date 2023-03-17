import math
import random

import torch

from debugger.config_data_classes.rl_checkers.agent_config import AgentConfig
from debugger.debugger_interface import DebuggerInterface
import torch.nn.functional as F


class AgentCheck(DebuggerInterface):
    """
    This class performs checks on the neural networks composing the agent, ensuring they are being updated and
    interacting correctly.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
            * old_target_model_params: a dictionary containing the last parameters of the target network before being
            updated.
            * old_training_data: a fixed batch of data used to measure the KL divergence. This data can be seen as
            the benchmarking data.
            * old_model_output: the main model's predictions on the old_training_data before being updated. This is
            useful to measure the KL divergence.
        """
        super().__init__(check_type="Agent", config=AgentConfig)
        self.old_target_model_params = None
        self.old_training_data = None
        self.old_model_output = None

    def run(
        self,
        model,
        target_model,
        actions_probs,
        target_model_update_period,
        training_observations,
        target_net_update_fraction=1,
    ) -> None:
        """
        I. This class checks whether the agent's neural networks are being updated and coordinated correctly. The
        agent typically consists of a main and target network, with the latter being a copy of the former. The two
        networks are updated periodically at different intervals to enhance the stability and efficiency of the
        learning process.
            * This class ensures that the target network is updated correctly when it reaches the update period ,
              otherwise has fixed parameters in the other steps. For the main network's parameters, it's essential to
              verify that it has parameters different from the target network's parameters, when it's not the target
              network's update period. In addition, it is also important to check that the main network is used to
              predict the next action and not hte target network, as using the target network can cause an unstable
              learning process.
            * Additionally, this class checks that the main network's updates are smooth and do not result in
              catastrophic forgetting, which can lead to an unstable and unreliable learning process.

        Note that the update of the target model can be achieved through one of two forms: hard update or soft update.
            * In a hard update, the weights of the target network are directly copied from the main network.
            * In a soft update, the weights of the target network are gradually updated over time by interpolating
              between the weights of the target network and the main network.

        II. This class is responsible for performing several checks on the agent's behavior during the learning
            process. It does the following checks :
            (1) Verifies that the target model has been updated at the correct period with the correct parameter values.
            (2) Verifies that the main and target model have different parameter values when it's not the target
                network's update period.
            (3) Verifies that the agent's predictions are being made by the correct model.
            (4) Detects significant divergence in the model's behavior between successive updates, which can indicate
                learning instability.

        III. The potential root causes behind the warnings that can be detected are
            - Incorrect network updates:
                * Target network not updated at the correct period (checks triggered: 1, 2)
                * Target network not updated with the values of the main network (whether it's a hard or soft update)
                  (checks triggered: 1)
            - Confusion between the main and target networks (checks triggered: 1, 2, 3)
            - Unstable learning process (checks triggered: 4)
            - Coding errors (checks triggered: 1, 2, 3)

        IV. The recommended fixes for the detected issues :
            - Check whether the target model is being updated at the correct time (checks that can be fixed: 1,2,3).
            - Check if you haven't mixed the main and target network:
                - You are using the main model to predict the next action (checks that can be fixed: 3)
                - You are updating the target network with the parameters of the main network (checks that can be fixed: 3)
            - Make sure the main model is updated correctly (checks that can be fixed: 4).
            - Check the models' hyperparameters to ensure that they are appropriately set (checks that can be fixed: 4).
            - Consider changing the update period of the target network to address learning instability issues (checks that can be fixed: 4).

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
        target_params = target_model.state_dict()
        current_params = model.state_dict()
        if self.old_target_model_params is None:
            self.old_target_model_params = target_params
            self.old_training_data = training_observations
        self.check_main_target_models_behaviour(
            target_params,
            current_params,
            target_net_update_fraction,
            target_model_update_period,
        )

        if self.check_period():
            self.check_wrong_model_output(model, training_observations, actions_probs)
            self.check_kl_divergence(model)
        self.old_model_output = model(self.old_training_data)

    def check_main_target_models_behaviour(
        self,
        target_params,
        current_params,
        target_net_update_fraction,
        target_model_update_period,
    ):
        """
        Checks whether the main and target models are being updated correctly during the learning process. This function
        verifies that the target model has been updated correctly when reaching the update period. It also checks
        whether both the main and target networks have different parameters during training.

        Args:
            target_params (dict): the weights and biases of the target model.
            current_params (dict): the weights and biases of the main model.
            target_net_update_fraction (float): The fraction of the target model parameters to be updated in each target model update.
            target_model_update_period (float):  The period of the target network update, (given as the number of steps taken).

        Returns:

        """
        random_layer_name = list(target_params.keys())[
            random.randint(2, len(target_params.keys()) - 2)
        ]
        all_equal = torch.equal(
            target_params[random_layer_name],
            (1 - target_net_update_fraction)
            * self.old_target_model_params[random_layer_name]
            + target_net_update_fraction * current_params[random_layer_name],
        )

        if (
            (((self.step_num - 1) % target_model_update_period) == 0)
            and (self.step_num > 1)
            and not self.config.target_update.disabled
        ):
            if not all_equal:
                self.error_msg.append(self.main_msgs["target_network_not_updated"])
            self.old_target_model_params = target_params

        else:
            if not self.config.similarity.disabled:
                if all_equal:
                    self.error_msg.append(
                        self.main_msgs["similar_target_and_main_network"]
                    )
                if not torch.equal(
                    self.old_target_model_params[random_layer_name],
                    target_params[random_layer_name],
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
        pred_qvals = model(observations)
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
            new_model_output = model(self.old_training_data)
            if not torch.allclose(
                torch.sum(new_model_output, dim=1),
                torch.ones(new_model_output.shape[0], device=self.device),
            ):
                new_model_output = F.softmax(new_model_output, dim=1)
                self.old_model_output = F.softmax(self.old_model_output, dim=1)
            kl_div = F.kl_div(
                torch.log(new_model_output),
                self.old_model_output,
                reduction="batchmean",
            )
            if torch.any(kl_div > self.config.kl_div.div_threshold):
                self.error_msg.append(
                    self.main_msgs["kl_div_high"].format(
                        kl_div, self.config.kl_div.div_threshold
                    )
                )
