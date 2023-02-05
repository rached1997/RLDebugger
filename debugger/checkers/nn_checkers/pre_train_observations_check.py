from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import almost_equal
import torch


def get_config():
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {"Period": 0,
              "Data": {"normalized_data_mins": [0.0, -1.0], "normalized_data_maxs": [1.0],
                       "labels_perp_min_thresh": 0.5, "outputs_var_coef_thresh": 0.001}
              }
    return config


class PreTrainObservationsCheck(DebuggerInterface):

    def __init__(self):
        super().__init__(check_type="PreTrainObservation", config=get_config())

    def run(self, observations):
        """
        Perform checks on the observations to ensure data quality.
        This function checks:
            1. If the observations are changing or if they all have the same value.
            2. If the observations are normalized.Normalization is highly recommended, as stated in the
             `Stable Baselines` documentation (https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment).

        Args:
            observations (Tensor): A collection of observations gathered prior to the model's training.
        """
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
