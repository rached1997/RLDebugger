import torch
from debugger.debugger_interface import DebuggerInterface
import numpy as np


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 15,
        "exploration_perc": 0.2,
        "vars_to_check": 10,
        "window_size": 5,
        "fluctuation": {"disabled": False, "fluctuation_rmse_min": 0.1},
        "monotonicity": {"disabled": False, "stagnation_thresh": 1e-3, "reward_stagnation_tolerance": 0.01,
                         "stagnation_episodes": 20}
    }
    return config


class OnTrainRewardsCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainReward", config=get_config())
        self.episodes_rewards = []

    def run(self, reward, max_total_steps, max_reward) -> None:

        if self.is_final_step_of_ep():
            self.episodes_rewards += [reward]

        n_rewards = len(self.episodes_rewards)
        if self.check_period() and (n_rewards >= self.config['window_size'] * self.config["vars_to_check"]):
            variances = []

            for i in range(0, len(self.episodes_rewards), self.config['window_size']):
                variances += [np.var(self.episodes_rewards[i:i + self.config['window_size']])]

            variances = torch.tensor(variances).float().reshape(-1, 1)
            cof = self.get_reward_var_slope(variances)

            if (self.step_num < max_total_steps * self.config['exploration_perc']) and \
                    (not self.config["fluctuation"]["disabled"]):
                fluctuations = self.check_reward_start_fluctuating(variances, cof)
                if fluctuations < self.config["fluctuation"]['fluctuation_rmse_min']:
                    self.error_msg.append(self.main_msgs['fluctuated_reward'].format(
                        self.config['exploration_perc'] * 100))

            if self.step_num > max_total_steps * (1 - self.config['exploration_perc']) and \
                    (not self.config["monotonicity"]["disabled"]):
                self.check_reward_monotonicity(cof, max_reward)

            self.episodes_rewards = []

    def check_reward_monotonicity(self, cof, max_reward):
        """
        Check if the entropy is increasing with time, or is stagnated.

        entropy_slope (float): The slope of the linear regression fit to the entropy values.
        :return: A warning message if the entropy is increasing or stagnated with time.
        """
        # TODO: debug this please cof.solution[0][0]
        if torch.abs(cof.solution[0][0]) > self.config["monotonicity"]["stagnation_thresh"]:
            self.error_msg.append(
                self.main_msgs['decreasing_reward'].format(100 - (self.config["exploration_perc"] * 100)))
        else:
            if self.episodes_rewards[self.config["monotonicity"]["stagnation_episodes"]:]:
                stagnated_reward = np.mean(self.episodes_rewards[self.config["monotonicity"]["stagnation_episodes"]:])
                if stagnated_reward < max_reward * self.config["monotonicity"]["reward_stagnation_tolerance"]:
                    self.error_msg.append(
                        self.main_msgs['stagnated_reward'].format(
                            100 - (self.config["exploration_perc"] * 100),
                            stagnated_reward,
                            max_reward))
        return None

    # TODO: debug this please
    def check_reward_start_fluctuating(self, vars, cof):
        x = torch.arange(len(vars), device=vars.device)
        ones = torch.ones_like(x)
        X = torch.stack([x, ones], dim=1).float()
        predicted = X.mm(cof.solution.T)

        residuals = torch.sqrt(torch.mean((vars - predicted) ** 2))
        return residuals

    def get_reward_var_slope(self, vars):
        """Compute the slope of rewards variance evolution over time.

        Returns:
        reward_variance_slope_coefficients (float): The slope of the linear regression fit to the rewards variance
        values.
        """
        x = torch.arange(len(vars))
        ones = torch.ones_like(x)
        X = torch.stack([x, ones], dim=1).float()
        cof = torch.linalg.lstsq(vars, X)
        return cof
