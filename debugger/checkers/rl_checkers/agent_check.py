import math
import random

import torch

from debugger.config_data_classes.rl_checkers.agent_config import AgentConfig
from debugger.debugger_interface import DebuggerInterface
import torch.nn.functional as F


class AgentCheck(DebuggerInterface):
    def __init__(self):
        """
        Initilizes the following parameters:
        old_target_model_params :it's a dict saving the last parameters of the target network before being updated
        old_training_data : It a fix batch of data we use to measure the kl divergence. This data can be seen as the benchmarking data
        old_model_output : the main model's predictions on the old_training_data before being updated. This is usefull  to measure the KL divergence
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
        This class checks if the agent's neural networks are being updated and coordinated correctly. By agent's
        neural network we refer to the main and target network. This class will check if the two networks are being
        updated and used correctly. In many DRL applications the agent can  be composed of a main and target network
        (which is a copy of the main network), being updated in different periods. The main reason behind using the
        two different networks in DRL is to improve the stability and efficiency of the learning process. Thus the
        normal behaviour of the agent should consist of periodically updating the target netwok with the values of
        the main network. Other than the update period the main and target network should have different parameters,
        and the target network parameters should be fix. In addition, it's essential to check that the network being
        used to predict the next action is the main network, otherwise, there will be an unstable learning process.
        In addition, when updating the main network it's essential to make sure that the main network's update is
        smooth and there is no catastrophic forgetting in the agent's behaviour, to ensure a stable and robust learning.

        Note, that the update of the target model can have two forms:
            - Hard update :  the weights of the target network are copied directly from the main network
            - Soft update : the weights of the target network are updated gradually over time by interpolating
            between the weights of the target network and the main network.

        The agent check does multiple checks on the agent's behaviour including:
        (1) Verifies that the target model has been updated at the correct time and with the correct parameter values.
        (2) Checks that the main and target model have different parameter values (when it's not the target network's update period).
        (3) Checks whether the agent's predictions are being made by the correct model.
        (4) Detects significant divergence in the model's behavior between successive updates, which can indicate learning instability.

        The potential root causes behind the warnings that can be detected are
            - A wrong network update :
                - The target network is not updated in the wrong period (checks triggered : 1,2)
                - The target network is not updated with the values of the main network (it can be a hard or a soft update) (checks triggered : 1)
            - Mixing up between the main and target networks (checks triggered : 1,2,3)
            - An unstable learning process (checks triggered : 4)
            - Coding errors (checks triggered : 1,2,3)

        The recommended fixes for the detected issues :
            - Check whether you are updating the target model in the right period (checks that can be fixed: 1,2,3)
            - Check if you haven't mixed the main and target network:
                - You are using the main model to predict the next action (checks that can be fixed: 3)
                - You are updating the target network with the parameters of the main network (checks that can be fixed: 3)
            - Check if you are updating the main model correctly (checks that can be fixed: 4)
            - Check the models hyperparameters (checks that can be fixed: 4)
            - Change the update period of the target network (checks that can be fixed: 4)

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
            and not self.config["target_update"]["disabled"]
        ):
            if not all_equal:
                self.error_msg.append(self.main_msgs["target_network_not_updated"])
            self.old_target_model_params = target_params

        else:
            if not self.config["similarity"]["disabled"]:
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
        if self.config["wrong_model_out"]["disabled"]:
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
        if self.iter_num > 1 and (not self.config["kl_div"]["disabled"]):
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
            if torch.any(kl_div > self.config["kl_div"]["div_threshold"]):
                self.error_msg.append(
                    self.main_msgs["kl_div_high"].format(
                        kl_div, self.config["kl_div"]["div_threshold"]
                    )
                )
