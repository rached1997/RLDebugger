import torch
from debugger.debugger_interface import DebuggerInterface
import numpy as np

from debugger.utils.utils import get_data_slope, estimate_fluctuation_rmse


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "period": 100,
        "exploration_perc": 0.2,
        "exploitation_perc": 0.8,
        "start": 5,
        "window_size": 3,
        "fluctuation": {"disabled": False, "fluctuation_rmse_min": 0.1},
        "monotonicity": {"disabled": False, "stagnation_thresh": 1e-3, "reward_stagnation_tolerance": 0.01,
                         "stagnation_episodes": 20}
    }
    return config


class RewardsCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="Reward", config=get_config())
        self.episodes_rewards = []

    def run(self, reward, max_total_steps, max_reward) -> None:
        """
        Does the following checks on the reward value :
        (1) checks if the reward per episode is fluctuating during the exploration
        (2) checks if the reward per episode is stagnating in the last episodes of the exploitation
        (3) checks whether the agent in the last episodes is stuck in a value far from the max reward expected

        Args:
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
        """

        if self.is_final_step():
            self.episodes_rewards += [reward]

        n_rewards = len(self.episodes_rewards)
        if self.check_period() and (n_rewards >= self.config['window_size'] * self.config["start"]):
            variances = []

            for i in range(0, len(self.episodes_rewards) // self.config['window_size']):
                count = i * self.config['window_size']
                variances += [np.var(self.episodes_rewards[count:count + self.config['window_size']])]

            variances = torch.tensor(variances).float()

            if (self.step_num < max_total_steps * self.config['exploration_perc']) and \
                    (not self.config["fluctuation"]["disabled"]):
                cof = get_data_slope(variances)
                fluctuations = estimate_fluctuation_rmse(cof, variances)
                if fluctuations < self.config["fluctuation"]['fluctuation_rmse_min']:
                    self.error_msg.append(self.main_msgs['fluctuated_reward'].format(
                        self.config['exploration_perc'] * 100))

            if self.step_num > max_total_steps * (self.config['exploitation_perc']) and \
                    (not self.config["monotonicity"]["disabled"]):
                cof = get_data_slope(variances)
                self.check_reward_monotonicity(cof, max_reward)

            self.episodes_rewards = []

    def check_reward_monotonicity(self, cof, max_reward):
        """
        Check if the variance of the reward is not stagnating in the last episodes. or if the reward is stuck in a
        value far from the max reward

        Args:
            cof (tuple): The slope of the linear regression fit to the variance values.
            max_reward (int):  The reward threshold before the task is considered solved
        """
        if torch.abs(cof[0]) > self.config["monotonicity"]["stagnation_thresh"]:
            self.error_msg.append(
                self.main_msgs['decreasing_reward'].format(100 - (self.config["exploration_perc"] * 100)))
        else:
            stagnated_reward = np.mean(self.episodes_rewards)
            if stagnated_reward < max_reward * (1- self.config["monotonicity"]["reward_stagnation_tolerance"]):
                self.error_msg.append(
                    self.main_msgs['stagnated_reward'].format(
                        100 - (self.config["exploration_perc"] * 100),
                        stagnated_reward,
                        max_reward))
        return None
