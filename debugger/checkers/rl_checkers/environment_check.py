import copy
import gym
import torch

from debugger.config_data_classes.rl_checkers.environment_config import EnvironmentConfig
from debugger.debugger_interface import DebuggerInterface
import numbers


class EnvironmentCheck(DebuggerInterface):
    """
    This class performs checks on the environment to ensure that it was designed correctly. These checks are mainly
    useful when you implement your new custom checks

    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
            * obs_list : A list of observations collected in a random episode
            * reward_list : A list of rewards collected in a random episode
            * done_list : A list of done flag collected in a random episode
        """
        super().__init__(check_type="Environment", config=EnvironmentConfig)
        self.obs_list = torch.tensor([])
        self.reward_list = torch.tensor([])
        self.done_list = torch.tensor([])

    def run(self, environment) -> None:
        """
        I. The EnvironmentChecks class is designed to verify the proper implementation of a custom environment. It is
        particularly useful when creating a new environment from scratch, rather than using a predefined one. The
        purpose of this class is to ensure that the environment adheres to the predefined rules, and to identify any
        issues before the training begins.
        To ensure the environment works as intended, two main functions are tested to check the behaviour of the
        environment:
            * The reset() function and the step() function. The reset() function should return an initial
            state and is used at the beginning of each episode to initialise it.
            * The step() function should return three essential elements: the state, the reward, and done a boolean
            indicating if the episode is finished, and other variables, if applicable.

        II. Before beginning the learning process, the environment check performs the following checks in the training
        environment to evaluate the environment's relevancy:
            (1) Verifies the environment's conception: verifies various features required in any DRL environment
                a. Verifies that observations and actions are bounded within a valid range.
                b. Checks that the step function returns valid observation values and does not produce NaN values.
                c. Ensures that the done flag is a boolean value and that the reward returned from the environment is
                   numerical.
                d. Verifies that the max_episode_steps parameter is numerical and sets a valid upper limit on episode
                   length.
                e. Checks that the max_reward parameter is numerical and sets a valid threshold for solving the task.
                f. Verifies that the reset function returns None to indicate that the environment is ready to start a
                   new episode.

            (2) Checks if the max reward threshold is too low
            (3) Checks whether the reward value is normalized

        III. The potential root causes behind the warnings that can be detected are
            - A bad conception of the environment (checks triggered : 1,2,3)
            - Bad hyperparameters of the environment (checks triggered : 2)
            - The environment is too easy to solve (checks triggered : 2)
            - Lack of preprocessing of the reward returned (checks triggered : 3)

        IV. The recommended fixes for the detected issues :
            - Check if the step function is coded correctly (checks that can be fixed: 1,2,3)
            - Check if the reset function is coded correctly (checks that can be fixed: 1)
            - Check the hyperparameters of the environment (checks that can be fixed: 2)
            - Check if the reward is returned correctly (checks that can be fixed: 3)


        Args:
            environment (gym.env): the training RL environment
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

                self.check_normalized_rewards()

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
            - Ensures that the done flag is a boolean value and that the reward returned from the environment is
              numerical.
            - Verifies that the max_episode_steps parameter is numerical and sets a valid upper limit on episode length.
            - Checks that the max_reward parameter is numerical and sets a valid threshold for solving the task.
            - Verifies that the reset function returns None to indicate that the environment is ready to start a new
              episode.

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

    def check_normalized_rewards(self):
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
