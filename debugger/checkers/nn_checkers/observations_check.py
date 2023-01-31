from debugger.debugger_interface import DebuggerInterface
from debugger.utils.metrics import almost_equal
import torch


def get_config():
    config = {"Period": 0,
              "Data": {"normalized_data_mins": [0.0, -1.0], "normalized_data_maxs": [1.0],
                       "labels_perp_min_thresh": 0.5, "outputs_var_coef_thresh": 0.001}
              }
    return config


class ObservationsCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="Observation", config=get_config())

    def run(self, observations):
        error_msg = list()
        mas = torch.max(observations)
        mis = torch.min(observations)
        avgs = torch.mean(observations * 1.0)
        stds = torch.std(observations * 1.0)

        if stds == 0.0:
            error_msg.append(self.main_msgs['features_constant'])
        elif any([almost_equal(mas, data_max) for data_max in self.config["Data"]["normalized_data_maxs"]]) and \
                any([almost_equal(mis, data_min) for data_min in self.config["Data"]["normalized_data_mins"]]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            error_msg.append(self.main_msgs['features_unnormalized'])
        return error_msg
