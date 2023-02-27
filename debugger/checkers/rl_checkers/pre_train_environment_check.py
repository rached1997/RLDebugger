import hashlib
import numpy as np
import torch
from hive.envs import GymEnv

from debugger.debugger_interface import DebuggerInterface
import gym


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 0,
        "observations_std_coef_thresh": 0.001,
        "Markovianity_check": {
            "disabled": False,
            "num_trajectories": 1000
        }
    }
    return config


class PreTrainEnvironmentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="PreTrainEnvironment", config=get_config())

    def generate_random_trajectories(self, env):
        trajectories = []
        for i in range(self.config["Markovianity_check"]["num_trajectories"]):
            obs = env.reset()
            trajectory = []
            for t in range(env.spec.max_episode_steps):
                action = env.action_space.sample()
                obs_next, reward, done, info = env.step(action)
                hashed_obs = str(hashlib.sha256(obs.tobytes()).hexdigest())
                hashed_obs_next = str(hashlib.sha256(obs_next.tobytes()).hexdigest())
                trajectory.append((hashed_obs, action, reward, hashed_obs_next))
                if done:
                    break
                obs = obs_next
            trajectories.append(trajectory)
        return trajectories

    def check_markovianity(self, env):
        trajectories = self.generate_random_trajectories(env)
        is_markovian = True
        for trajectory in trajectories:
            for t in range(len(trajectory) - 1):
                obs_t, action_t, reward_t, obs_next_t = trajectory[t]
                obs_next_t_predicted = env.reset()
                for t_prime in range(t, len(trajectory)):
                    obs_t_prime, action_t_prime, reward_t_prime, obs_next_t_prime = trajectory[t_prime]
                    if np.array_equal(obs_t_prime, obs_t):
                        obs_next_t_predicted = obs_next_t_prime
                        break
                if not np.array_equal(obs_next_t, obs_next_t_predicted):
                    is_markovian = False
                    break
            if not is_markovian:
                break

    def run(self, environment: gym.envs) -> None:
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        done = False
        initial_obs = environment.reset()

        step = 0
        while (not done) and (step < environment.spec.max_episode_steps):
            step += 1
            obs, reward, done, info = environment.step(environment.action_space.sample())
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        if sum(reward_list) > environment.spec.reward_threshold:
            self.error_msg.append(self.main_msgs['Weak_reward_threshold'])

        if np.std(obs_list) <= self.config["observations_std_coef_thresh"]:
            self.error_msg.append(self.main_msgs['invalid_step_func'].format(np.var(obs_list)))

        if not self.config["Markovianity_check"]["disabled"]:
            self.check_markovianity(environment)

        # todo check again
        # if (step >= environment.spec.max_episode_steps) and (not done):
        #     self.error_msg.append(self.main_msgs['missing_terminal_state'])


        # todo check again
        # initial_obs_test = environment.reset()
        # if (not self.config["reset_func_check"]["disabled"]) and (initial_obs_test != initial_obs).all():
        #     self.error_msg.append(self.main_msgs['non_deterministic_reset_function'])


