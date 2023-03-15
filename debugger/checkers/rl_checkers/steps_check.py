import statistics

import numpy as np
import torch

from debugger.config_data_classes.rl_checkers.steps_config import StepsConfig
from debugger.debugger_interface import DebuggerInterface


class StepCheck(DebuggerInterface):
    def __init__(self):
        """
        Initializes the following parameters:
        final_step_number_buffer :  list storing the number of steps taken in each episode
        episode_reward_buffer : lis storing the rewards accumulated in each episode
        last_step_num : an in meantioning in the previous episode what was the totla number of steps the agent did
        """
        super().__init__(check_type="Step", config=StepsConfig)
        self.final_step_number_buffer = []
        self.episode_reward_buffer = []
        self.last_step_num = 0

    def run(self, reward, max_reward, max_total_steps, max_steps_per_episode) -> None:
        """
        Checks if episodes are being ended prematurely due to the max step limit being reached during the
        exploitation phase when the agent is not learning (i.e. the reward is far from the max reward).

        The steps check performs the following check:
            (1) Checks whether the max steps per episode has a low value.

        The potential root causes behind the warnings that can be detected are:
            - The max steps per episode has a low value (checks triggered : 1)

        The recommended fixes for the detected issues :
            - Increase the max stepsper episode value (checks that can be fixed: 1)

        Args:
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
            max_steps_per_episode (int): the max steps for an episode
        """
        if self.is_final_step():
            self.final_step_number_buffer += [self.step_num - self.last_step_num]
            self.episode_reward_buffer += [reward]
            self.last_step_num = self.step_num

        self.check_step_is_not_changing(
            max_reward, max_total_steps, max_steps_per_episode
        )

    def check_step_is_not_changing(
        self, max_reward, max_total_steps, max_steps_per_episode
    ):
        """
        Checks if episodes are being ended prematurely due to the max step limit being reached during the
        exploitation phase when the agent is not learning (i.e. the reward is far from the max reward).

        Args:
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
            max_steps_per_episode (int): the max steps for an episode
        """
        if self.config.check_stagnation.disabled:
            return

        if self.check_period() and (
            self.step_num >= (max_total_steps * self.config.exploitation_perc)
        ):
            if (
                statistics.mean(self.final_step_number_buffer) >= max_steps_per_episode
            ) and (
                statistics.mean(self.episode_reward_buffer)
                < (max_reward * self.config.poor_max_step_per_ep.max_reward_tol)
            ):
                self.error_msg.append(self.main_msgs["poor_max_step_per_ep"])
