import torch

from debugger.config_data_classes.rl_checkers.reward_config import RewardConfig
from debugger.debugger_interface import DebuggerInterface

from debugger.utils.utils import get_data_slope, estimate_fluctuation_rmse


class RewardsCheck(DebuggerInterface):
    """
    This class performs checks on the accumulated reward.
    For more details on the specific checks performed, refer to the `run()` function.
    """
    def __init__(self):
        """
        Initializes the following parameters:
            * _episodes_rewards : The reward accumulated in each episode
        """
        super().__init__(check_type="Reward", config=RewardConfig)
        self._episodes_rewards = torch.tensor([], device=self.device)

    def run(self, reward, max_total_steps, max_reward) -> None:
        """
        -----------------------------------   I. Introduction of the Reward Check   -----------------------------------

        The Reward Check class evaluates the behavior of the reward obtained by a DRL agent during the training
        process. The goal of a DRL agent is to maximize the reward it receives from the environment by improving the
        precision of the actions it takes.
        The normal behavior of a reward is that, during the exploration phase, the agent receives rewards with
        variable magnitudes in each episode. As the agent learns, the accumulated reward becomes more stable,
        and the agent approaches reaching the maximum expected reward.
        This class uses the reward's standard deviation (std) to analyse the agent's accumulated reward behaviour.
        The use of std helps remove noise and outliers and smooth the reward plot. During the exploration the
        behavior of the DRL application is mostly dependent on random actions, thus, the std should ideally begin
        with fluctuating reward values. As the exploration phase reduces, the std of the accumulated reward
        stabilizes, and the maximum reward in the final episodes should be close to zero.
        This class helps identify issues related to the behavior of the reward and facilitates the monitoring and
        improvement of the DRL agent's training process.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        The reward check class does the following checks on the reward values accumulated in each episode :
            (1) Checks if the reward per episode is fluctuating during the exploration
            (2) Checks if the reward per episode is stagnating in the last episodes of the exploitation
            (3) Checks whether the agent in the last episodes is stuck in a value far from the max reward expected in
                the last episodes of training

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are :
            - Missing exploration (checks triggered : 1,2,3)
            - An unbalanced exploration-exploitation trade-off (checks triggered : 1,2,3)
            - Wrong max reward value (checks triggered : 3)
            - Bad conception of the environment (checks triggered : 1, 2, 3)
            - Bad agent's architecture (i.e. the agent is not learning correctly) (checks triggered : 1, 2, 3)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues :
            - Do more exploration (checks that can be fixed: 1,2,3)
            - Change the ratio of the exploration-exploitation (checks that can be fixed: 1,2,3)
            - Reduce the max_reward value if possible (checks that can be fixed: 3)
            - Change the architecture of the agent (checks that can be fixed: 1, 2, 3)
                - change its parameters
                - change the number of layers
            - Check if the step function of the environment is working correctly (checks that can be fixed: 1, 2, 3)

        Examples
        --------
        To perform reward checks, the debugger needs "max_total_steps" and "max_reward" parameters only (constant
        parameters). The reward parameter is automatically observed by debugger, and you don't need to pass the reward to
        the 'debug()' function.
        The best way to run these checks is to provide the "max_total_steps" and "max_reward" at the begging of your
        code.

        >>> from debugger import rl_debugger
        >>> ...
        >>> env = gym.make("CartPole-v1")
        >>> rl_debugger.debug(max_reward=max_reward, max_total_steps=max_total_steps)

        If you feel that these checks are slowing your code, you can increase the value of "skip_run_threshold" in
        RewardConfig.

        Args:
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
        """
        if self.is_final_step():
            self._episodes_rewards = torch.cat(
                (
                    self._episodes_rewards,
                    torch.tensor(reward, device=self.device).view(1),
                ),
                dim=0,
            )

        if self.skip_run(self.config.skip_run_threshold):
            return
        n_rewards = len(self._episodes_rewards)
        if self.check_period() and (
            n_rewards >= self.config.window_size * self.config.start
        ):
            stds = []
            stds_nor = []

            for i in range(0, len(self._episodes_rewards) // self.config.window_size):
                count = i * self.config.window_size
                reward_std = torch.std(
                    self._episodes_rewards[count: count + self.config.window_size]
                )
                reward_std_nor = torch.std(
                    self._episodes_rewards[count: count + self.config.window_size]
                    / max_reward
                )
                self.wandb_metrics = {
                    "reward_stds": reward_std
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

            self._episodes_rewards = torch.tensor([], device=self.device)

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
            stagnated_reward = torch.mean(self._episodes_rewards)
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
