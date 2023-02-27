import torch
from debugger.debugger_interface import DebuggerInterface
import numpy as np

from debugger.utils.utils import smoothness


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 15,
        "disabled": False,
        "episodes_to_check" : 10,
        "window_size": 5,
        "incr_percentage": 0.05,
        "exploration_perc": 0.2,
        "stagnation_thresh": 1e-3,
        "reward_stagnation_tolerance": 1e-3,
        "stagnation_episodes": 20,
        "fluctuation": 0.1}
    return config


class OnTrainRewardsCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainReward", config=get_config())
        self.total_steps = 0

        self.episodes_rewards = []

    def run(self, episode_reward, done, steps, max_steps_per_episode, max_total_steps, max_reward) -> None:

        # self.episodes_rewards.append(episode_reward)
        self.total_steps += steps

        if done or steps >= max_steps_per_episode:
            self.episodes_rewards += [episode_reward]

        n_rewards = len(self.episodes_rewards)
        if self.check_period() and (n_rewards >= self.config['window_size'] * self.config["episodes_to_check"] ):
            vars = []

            for i in range(0, len(self.episodes_rewards), self.config['window_size']):
                vars += [np.var(self.episodes_rewards[i:i + self.config['window_size']])]

            vars = torch.tensor(vars).float().reshape(-1, 1)
            cof = self.get_reward_var_slope(vars)

            if self.iter_num < max_total_steps * self.config['exploration_perc']:
                fluctuations = self.check_reward_start_fluctuating(vars, cof)
                if fluctuations < self.config['fluctuation']:
                    self.error_msg.append(self.main_msgs['fluctuated_reward'].format(
                        self.config['exploration_perc']*100))

            if self.iter_num > max_total_steps * (1 - self.config['exploration_perc']):
                self.check_reward_monotonicity(cof, max_reward)

            self.episodes_rewards = []

    def check_reward_monotonicity(self, cof, max_reward):
        """
        Check if the entropy is increasing with time, or is stagnated.

        entropy_slope (float): The slope of the linear regression fit to the entropy values.
        :return: A warning message if the entropy is increasing or stagnated with time.
        """
        if torch.abs(cof.solution[0][0]) > self.config["stagnation_thresh"]:
            self.error_msg.append(
                self.main_msgs['decreasing_reward'].format(100- (self.config["exploration_perc"]*100)))
        else:
            if self.episodes_rewards[self.config["stagnation_episodes"]:]:
                stagnated_reward = np.mean(self.episodes_rewards[self.config["stagnation_episodes"]:])
                if stagnated_reward < max_reward* self.config["reward_stagnation_tolerance"]:
                    self.error_msg.append(
                        self.main_msgs['stagnated_reward'].format(
                            100 - (self.config["exploration_perc"] * 100),
                            stagnated_reward,
                            max_reward))
        return None

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
