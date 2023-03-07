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
        "Period": 0,
        "observations_std_coef_thresh": 0.001,
        "Markovianity_check": {"disabled": False, "num_trajectories": 1000}
    }
    return config


class PreTrainEnvironmentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="PreTrainEnvironment", config=get_config())
        self.obs_list = torch.tensor([])
        self.reward_list = torch.tensor([])
        self.done_list = torch.tensor([])

    def run(self, environment) -> None:

        # todo CODE: add Markovianity check
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
