import torch
from debugger.debugger_interface import DebuggerInterface
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
        "wrong_model_out": {"disabled": False}
    }
    return config


class OnTrainAgentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainAgent", config=get_config())
        self.old_target_params = None

    # todo DOC : we have to explain in the doc that target_net_update_fraction=1 when there is not a soft update
    # todo CR : check with darshan if the target and main network should be initialized with the same values
    def run(self, model, target_model, observation, actions_probs, target_model_update_period,
            target_net_update_fraction=1) -> None:
        self.check_main_target_models_behaviour(target_model, model, target_net_update_fraction,
                                                target_model_update_period)

        if self.check_period():
            self.check_wrong_model_output(model, observation, actions_probs)

    def check_main_target_models_behaviour(self, target_model, model, target_net_update_fraction,
                                           target_model_update_period):
        target_params = target_model.state_dict()
        current_params = model.state_dict()
        if self.old_target_params is None:
            self.old_target_params = target_params

        all_equal = all(torch.equal(target_params[key],
                                    (1 - target_net_update_fraction) * self.old_target_params[key] +
                                    target_net_update_fraction * current_params[key])
                        for key in target_params)

        if (0 == ((self.step_num - 1) % target_model_update_period)) and (self.step_num > 1) and \
                not self.config["target_update"]["disabled"]:

            if not all_equal:
                self.error_msg.append(self.main_msgs['target_network_not_updated'])
            self.old_target_params = target_params

        else:
            if all_equal and not self.config["similarity"]["disabled"]:
                self.error_msg.append(self.main_msgs['similar_target_and_main_network'])

    def check_wrong_model_output(self, model, observations, action_probs):
        if self.config["wrong_model_out"]["disabled"]:
            return
        pred_qvals = model(observations)
        if not torch.equal(action_probs, pred_qvals):
            self.error_msg.append(self.main_msgs['using_the_wrong_network'])
