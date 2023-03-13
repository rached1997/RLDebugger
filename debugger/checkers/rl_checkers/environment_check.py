import copy
import hashlib

import gym
import torch
from debugger.debugger_interface import DebuggerInterface
import numbers


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "period": 0,
        "observations_std_coef_thresh": 0.001,
        "Markovianity_check": {"disabled": False, "num_trajectories": 1000},
        "normalization": {"disabled": False, "normalized_reward_min": -10.0, "normalized_reward_max": 10.0},

    }
    return config


class EnvironmentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="Environment", config=get_config())
        self.obs_list = torch.tensor([])
        self.reward_list = torch.tensor([])
        self.done_list = torch.tensor([])

    def run(self, environment) -> None:
        """
        Does the following checks in the environment:
        (1) Checks the conception of the environment
        (2) Checks if the max reward threshold is too low
        (3) Checks whether the reward value is normalized

        Args:
            environment (gym.env): the training RL environment

        Returns:

        """
        # todo IDEA: add Markovianity check
        if self.check_period():
            if environment.spec.max_episode_steps:
                self.generate_random_eps(environment)
                self.check_env_conception(environment)
                if sum(self.reward_list) > environment.spec.reward_threshold:
                    self.error_msg.append(self.main_msgs['Weak_reward_threshold'])

                if torch.mean(torch.std(self.obs_list, dim=0)) <= self.config["observations_std_coef_thresh"]:
                    self.error_msg.append(
                        self.main_msgs['invalid_step_func'].format(torch.mean(torch.std(self.obs_list, dim=0))))

    def generate_random_eps(self, environment):
        """
        Generate a random episode.
        Args:
            environment (gym.env): the training RL environment

        Returns:

        """
        environment = copy.deepcopy(environment)
        done = False
        initial_obs = torch.tensor(environment.reset())
        self.obs_list = torch.cat((self.obs_list, initial_obs), dim=0)

        step = 0
        while (not done) and (step < environment.spec.max_episode_steps):
            step += 1
            obs, reward, done, info = environment.step(environment.action_space.sample())
            self.obs_list = torch.cat((self.obs_list, torch.tensor(obs)), dim=0)
            self.reward_list = torch.cat((self.reward_list, torch.tensor([reward])), dim=0)
            self.done_list = torch.cat((self.done_list, torch.tensor([done])), dim=0)

    def check_env_conception(self, env: gym.envs):
        """
        Performs several checks on the agent's behavior to ensure its proper conception:
            - Verifies that observations and actions are bounded within a valid range.
            - Checks that the step function returns valid observation values and does not produce NaN values.
            - Ensures that the done flag is a boolean value and that the reward returned from the environment is numerical.
            - Verifies that the max_episode_steps parameter is numerical and sets a valid upper limit on episode length.
            - Checks that the max_reward parameter is numerical and sets a valid threshold for solving the task.
            - Verifies that the reset function returns None to indicate that the environment is ready to start a new episode.

        Args:
            env (gym.env): the training RL environment
        """
        def is_numerical(x):
            return isinstance(None, numbers.Number) and (x is not torch.inf) and (x is not torch.nan)

        if not (isinstance(env.observation_space, gym.spaces.Box) or isinstance(env.observation_space, gym.spaces.Discrete)):
            self.error_msg.append(self.main_msgs['bounded_observations'])

        if not (isinstance(env.action_space, gym.spaces.Box) or (isinstance(env.action_space, gym.spaces.Discrete))):
            self.error_msg.append(self.main_msgs['bounded_actions'])

        if torch.any(torch.isnan(self.obs_list)):
            self.error_msg.append(self.main_msgs['observation_not_returned'])

        if not all((isinstance(b, bool) or (b in [0,1])) for b in self.done_list):
            self.error_msg.append(self.main_msgs['non_bool_done'])

        if all(is_numerical(r) for r in self.reward_list):
            self.error_msg.append(self.main_msgs['reward_not_numerical'])

        if is_numerical(env.spec.max_episode_steps):
            self.error_msg.append(self.main_msgs['max_episode_steps_not_numerical'])

        if is_numerical(env.spec.reward_threshold):
            self.error_msg.append(self.main_msgs['reward_threshold_not_numerical'])

        if env.reset() is None:
            self.error_msg.append(self.main_msgs['wrong_reset_func'])

    def check_normalized_rewards(self, reward):
        """
        Check if the reward value is normalized.

        Args:
            reward (float): the reward returned at each step

        """
        if self.config["normalization"]["disabled"]:
            return

        max_reward_value = self.config["normalization"]["normalized_reward_max"]
        min_reward_value = self.config["normalization"]["normalized_reward_min"]

        if torch.max(self.reward_list) > max_reward_value or torch.min(self.reward_list) < min_reward_value:
            self.error_msg.append(self.main_msgs['reward_unnormalized'])
