from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import almost_equal
import torch


def get_config():
    config = {"Period": 0,
              "Data": {"normalized_data_mins": [0.0, -1.0], "normalized_data_maxs": [1.0],
                       "labels_perp_min_thresh": 0.5, "outputs_var_coef_thresh": 0.001}
              }
    return config


class PreTrainObservationsCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="PreTrainObservation", config=get_config())

    def run(self, observations):
        if not self.check_period():
            return
        mas = torch.max(observations)
        mis = torch.min(observations)
        avgs = torch.mean(observations * 1.0)
        stds = torch.std(observations * 1.0)

        if stds == 0.0:
            self.error_msg.append(self.main_msgs['features_constant'])
        # TODO: This function should work when we send the enviroenemnt as an argument
        elif any([(mas < data_max) for data_max in self.config["Data"]["normalized_data_maxs"]]) and \
                any([(mis < data_min) for data_min in self.config["Data"]["normalized_data_mins"]]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            self.error_msg.append(self.main_msgs['features_unnormalized'])
