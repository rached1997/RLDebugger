import torch
from debugger.debugger_interface import DebuggerInterface
import torch.nn.functional as F
import hashlib


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 100,
        "target_update": {"disabled": False},
        "similarity": {"disabled": False},
        "wrong_model_out": {"disabled": False},
        "kl_div": {"disabled": False, "div_threshold":0.1}
    }
    return config


class OnTrainAgentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainAgent", config=get_config())
        self.old_target_model_params = None
        self.old_model_output = None
        self.old_training_data = None

    # todo DOC : we have to explain in the doc that target_net_update_fraction=1 when there is not a soft update
    # todo CR : check with darshan if the target and main network should be initialized with the same values
    def run(self, model, target_model, actions_probs, target_model_update_period, training_observations,
            target_net_update_fraction=1) -> None:
        target_params = target_model.state_dict()
        current_params = model.state_dict()
        if self.old_target_model_params is None:
            self.old_target_model_params = target_params
            self.old_training_data = training_observations
        self.check_main_target_models_behaviour(target_params, current_params, target_net_update_fraction,
                                                target_model_update_period)

        if self.check_period():
            self.check_wrong_model_output(model, training_observations, actions_probs)
            self.check_kl_divergence(model)
        self.old_model_output = model(self.old_training_data)

    def check_main_target_models_behaviour(self, target_params, current_params, target_net_update_fraction,
                                           target_model_update_period):

        all_equal = all(torch.equal(target_params[key],
                                    (1 - target_net_update_fraction) * self.old_target_model_params[key] +
                                    target_net_update_fraction * current_params[key])
                        for key in target_params)

        if (0 == ((self.step_num - 1) % target_model_update_period)) and (self.step_num > 1) and \
                not self.config["target_update"]["disabled"]:

            if not all_equal:
                self.error_msg.append(self.main_msgs['target_network_not_updated'])
            self.old_target_model_params = target_params

        else:
            if all_equal and not self.config["similarity"]["disabled"]:
                self.error_msg.append(self.main_msgs['similar_target_and_main_network'])

    def check_wrong_model_output(self, model, observations, action_probs):
        if self.config["wrong_model_out"]["disabled"]:
            return
        pred_qvals = model(observations)
        if not torch.equal(action_probs, pred_qvals):
            self.error_msg.append(self.main_msgs['using_the_wrong_network'])

    def check_kl_divergence(self, model):
        if self.iter_num > 1 and (not self.config["kl_div"]["disabled"]):
            new_model_output = model(self.old_training_data)
            if not torch.allclose(torch.sum(new_model_output, dim=1), torch.ones(new_model_output.shape[0],
                                                                                 device='cuda')):
                new_model_output = F.softmax(new_model_output, dim=1)
                self.old_model_output = F.softmax(self.old_model_output, dim=1)
            kl_div = F.kl_div(torch.log(new_model_output), self.old_model_output, reduction='batchmean')
            if torch.any(kl_div > self.config["kl_div"]["div_threshold"]):
                self.error_msg.append(self.main_msgs['kl_div_high'].format(kl_div,
                                                                           self.config["kl_div"]["div_threshold"]))


