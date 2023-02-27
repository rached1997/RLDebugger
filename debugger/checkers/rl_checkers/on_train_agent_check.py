import torch
from debugger.debugger_interface import DebuggerInterface


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 1, }
    return config


class OnTrainAgentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainAgent", config=get_config())

    # todo : we have to explain in the doc that target_net_update_fraction=1 when there is not a soft update
    # todo check with darshan if the target and main network should be initialized with the same values
    def run(self, model, target_model, target_model_update_period, steps, target_net_update_fraction=1,
            predictions=None, observations=None, actions=None) -> None:
        target_params = target_model.state_dict()
        current_params = model.state_dict()
        if (0 == ((steps - 1) % target_model_update_period)) and (steps > 1):
            for key in list(target_params.keys()):
                # TODO: fix the soft use case
                if not torch.equal(target_params[key],
                                   (1 - target_net_update_fraction) * target_params[key] + target_net_update_fraction *
                                   current_params[key]):
                    self.error_msg.append(self.main_msgs['target_network_not_updated'])
        else:
            # todo : optimize code bool = .... if true if false
            for key in list(target_params.keys()):
                if torch.equal(target_params[key],
                               (1 - target_net_update_fraction) * target_params[key] + target_net_update_fraction *
                               current_params[key]):
                    self.error_msg.append(self.main_msgs['similar_target_and_main_network'])
                else:
                    break

        # todo change it place to a check that uses pred, obs, act
        if (not (predictions is None)) and (not (observations is None)):
            pred_qvals = model(observations)
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]
            if not torch.equal(predictions, pred_qvals):
                self.error_msg.append(self.main_msgs['using_the_wrong_network'])
