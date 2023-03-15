import copy
import gym
import torch

from debugger.config_data_classes.rl_checkers.environment_config import EnvironmentConfig
from debugger.debugger_interface import DebuggerInterface
import numbers


class EnvironmentCheck(DebuggerInterface):
    def __init__(self):
        """
        Initializes the following parameters:
        obs_list : A list of observations collected in a random episode
        reward_list : A list of rewards collected in a random episode
        done_list : A list of done flag collected in a random episode
        """
        super().__init__(check_type="Environment", config=EnvironmentConfig)
        self.obs_list = torch.tensor([])
        self.reward_list = torch.tensor([])
        self.done_list = torch.tensor([])

    def run(self, environment) -> None:
        """
        The environment checks consists of verifying that environment was correctly implemented. This class is mainly
        usefull when you implement your own environment, in other words you are not using a predifined environement.
        The main goal of the function is to verify the good conception of the environment and making sure that it
        doesn't violate the multiple predefined rules. The checks are done once before starting the training to make
        sure that the environment would work well during the training. A normal environement is required to provide
        mainlytwo functions the reset and the steps function. The reset function should return a state, and the step
        function should return three essential elements, namely the state, the reward and done (ot can also return
        other variables).

        To evalaute the good conception of the environmnet, the environement check does the following checks in the
        training environment before starting the learning process:

            (1) Checks the conception of the environment: checks multiple features required in any DRL environment you
            can check the function check_env_conception for more details
            (2) Checks if the max reward threshold is too low
            (3) Checks whether the reward value is normalized

        The potential root causes behind the warnings that can be detected are
            - A bad conception of the environment (checks triggered : 1,2,3)
            - Bad hyperparameters of the environment (checks triggered : 2)
            - The environement is too easy to solve (checks triggered : 2)
            - Lack of preprocessing of the reward returned (checks triggered : 3)

        The recommended fixes for the detected issues :
            - Check if the step function is coded correctly (checks that can be fixed: 1,2,3)
            - Check if the reset function is coded correctly (checks that can be fixed: 1)
            - Check the hyperparameters of the environment (checks that can be fixed: 2)
            - Check if the reward is returned correctly (checks that can be fixed: 3)


        Args:
            environment (gym.env): the training RL environment

        Returns:

        """
        if self.check_period():
            if environment.spec.max_episode_steps:
                self.generate_random_eps(environment)
                self.check_env_conception(environment)
                if sum(self.reward_list) > environment.spec.reward_threshold:
                    self.error_msg.append(self.main_msgs['weak_reward_threshold'])

                if (
                    torch.mean(torch.std(self.obs_list, dim=0))
                    <= self.config.observations_std_coef_thresh
                ):
                    self.error_msg.append(
                        self.main_msgs["invalid_step_func"].format(
                            torch.mean(torch.std(self.obs_list, dim=0))
                        )
                    )

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
            obs, reward, done, info = environment.step(
                environment.action_space.sample()
            )
            self.obs_list = torch.cat((self.obs_list, torch.tensor(obs)), dim=0)
            self.reward_list = torch.cat(
                (self.reward_list, torch.tensor([reward])), dim=0
            )
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
            return (
                isinstance(None, numbers.Number)
                and (x is not torch.inf)
                and (x is not torch.nan)
            )

        if not (
            isinstance(env.observation_space, gym.spaces.Box)
            or isinstance(env.observation_space, gym.spaces.Discrete)
        ):
            self.error_msg.append(self.main_msgs["bounded_observations"])

        if not (
            isinstance(env.action_space, gym.spaces.Box)
            or (isinstance(env.action_space, gym.spaces.Discrete))
        ):
            self.error_msg.append(self.main_msgs["bounded_actions"])

        if torch.any(torch.isnan(self.obs_list)):
            self.error_msg.append(self.main_msgs["observation_not_returned"])

        if not all((isinstance(b, bool) or (b in [0, 1])) for b in self.done_list):
            self.error_msg.append(self.main_msgs["non_bool_done"])

        if all(is_numerical(r) for r in self.reward_list):
            self.error_msg.append(self.main_msgs["reward_not_numerical"])

        if is_numerical(env.spec.max_episode_steps):
            self.error_msg.append(self.main_msgs["max_episode_steps_not_numerical"])

        if is_numerical(env.spec.reward_threshold):
            self.error_msg.append(self.main_msgs["reward_threshold_not_numerical"])

        if env.reset() is None:
            self.error_msg.append(self.main_msgs["wrong_reset_func"])

    def check_normalized_rewards(self, reward):
        """
        Check if the reward value is normalized.

        Args:
            reward (float): the reward returned at each step

        """
        if self.config.normalization.disabled:
            return

        max_reward_value = self.config.normalization.normalized_reward_max
        min_reward_value = self.config.normalization.normalized_reward_min

        if (
            torch.max(self.reward_list) > max_reward_value
            or torch.min(self.reward_list) < min_reward_value
        ):
            self.error_msg.append(self.main_msgs["reward_unnormalized"])
