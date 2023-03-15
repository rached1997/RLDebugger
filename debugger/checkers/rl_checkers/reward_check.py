import torch

from debugger.config_data_classes.rl_checkers.reward_config import RewardConfig
from debugger.debugger_interface import DebuggerInterface
import numpy as np

from debugger.utils.utils import get_data_slope, estimate_fluctuation_rmse


class RewardsCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="Reward", config=RewardConfig)
        self.episodes_rewards = torch.tensor([], device=self.device)

    def run(self, reward, max_total_steps, max_reward) -> None:
        """
        This class checks if the behaviour of the reward is normal or not.  DRL agent's goal is  to maximize the
        reward it gets from the environment by improving the accuracy of the action it takes. The normal behaviour of
        a reward is that during the exploration the agent keeps on getting rewards with variable magnitude in each
        episode and the more it learns the more the accumulated reward becomes more stable and the agent approaches
        from reaching the maximum reward expected. Thus, in order to analyse the behaviour of the reward and to
        smoothen the reward plot this classe employs the standard deviation (std) of the reward to reduce the noise
        and the outliers. What is expected is that the std starts with fluctuating values during the exploration as
        the agent's behaviour is more based on random actions. The more the exploration is reduced the more the std
        of the accumulated reward stabilise and in the last episodes the max reward should be close to zero.

        The reward check class does the following checks on the reward values accumulated in each episode :
        (1) checks if the reward per episode is fluctuating during the exploration
        (2) checks if the reward per episode is stagnating in the last episodes of the exploitation
        (3) checks whether the agent in the last episodes is stuck in a value far from the max reward expected in the
        last episodes of training


        The potential root causes behind the warnings that can be detected are
            - Missing exploration (checks triggered : 1,2,3)
            - An unbalanced exploration-exploitation rate (checks triggered : 1,2,3)
            - Wrong max reward value (checks triggered : 3)
            - Bad conception of the environment (checks triggered : 1, 2, 3)
            - Bad agent's architecture (i.e the agent is not learning correctly) (checks triggered : 1, 2, 3)


        The recommended fixes for the detected issues :
            - Do more exploration (checks that can be fixed: 1,2,3)
            - Change the ration of the exploration-exploitation (checks that can be fixed: 1,2,3)
            - Reduce the max_reward value if possible (checks that can be fixed: 3)
            - Change the architecture of the agent (checks that can be fixed: 1, 2, 3)
                - change its parameters
                - change the number of layers
            - Check if the step function of the environment is working correctly (checks that can be fixed: 1, 2, 3)

        Args:
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
        """
        if self.is_final_step():
            self.episodes_rewards = torch.cat(
                (
                    self.episodes_rewards,
                    torch.tensor(reward, device=self.device).view(1),
                ),
                dim=0,
            )

        if self.skip_run(self.config.skip_run_threshold):
            return
        n_rewards = len(self.episodes_rewards)
        if self.check_period() and (
            n_rewards >= self.config.window_size * self.config.start
        ):
            stds = []
            stds_nor = []

            for i in range(0, len(self.episodes_rewards) // self.config.window_size):
                count = i * self.config.window_size
                reward_std = torch.std(
                    self.episodes_rewards[count : count + self.config.window_size]
                )
                reward_std_nor = torch.std(
                    self.episodes_rewards[count : count + self.config.window_size]
                    / max_reward
                )
                self.wandb_metrics = {
                    "reward_stds": reward_std,
                    "reward_stds_nor": reward_std_nor,
                }
                stds += [reward_std]
                stds_nor += [reward_std_nor]

            stds = torch.tensor(stds).float()
            stds_nor = torch.tensor(stds_nor).float()

            if (self.step_num < max_total_steps * self.config.exploration_perc) and (
                not self.config.fluctuation.disabled
            ):
                cof = get_data_slope(stds)
                fluctuations = estimate_fluctuation_rmse(cof, stds)
                if fluctuations < self.config.fluctuation.fluctuation_rmse_min:
                    self.error_msg.append(
                        self.main_msgs["fluctuated_reward"].format(
                            self.config.exploration_perc * 100
                        )
                    )

            if self.step_num > max_total_steps * (self.config.exploitation_perc) and (
                not self.config.monotonicity.disabled
            ):
                cof = get_data_slope(stds_nor)
                self.check_reward_monotonicity(cof, max_reward)

            self.episodes_rewards = torch.tensor([], device=self.device)

    def check_reward_monotonicity(self, cof, max_reward):
        """
        Check if the std of the reward is not stagnating in the last episodes. or if the reward is stuck in a
        value far from the max reward

        Args:
            cof (tuple): The slope of the linear regression fit to the std values.
            max_reward (int):  The reward threshold before the task is considered solved
        """
        if torch.abs(cof[0]) > self.config.monotonicity.stagnation_thresh:
            self.error_msg.append(
                self.main_msgs["decreasing_reward"].format(
                    100 - (self.config.exploration_perc * 100)
                )
            )
        else:
            stagnated_reward = torch.mean(self.episodes_rewards)
            if stagnated_reward < max_reward * (
                1 - self.config.monotonicity.reward_stagnation_tolerance
            ):
                self.error_msg.append(
                    self.main_msgs["stagnated_reward"].format(
                        100 - (self.config.exploration_perc * 100),
                        stagnated_reward,
                        max_reward,
                    )
                )
        return None
