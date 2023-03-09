import hashlib
import statistics

import torch
from debugger.debugger_interface import DebuggerInterface


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 1,
        "exploitation_perc": 0.8,
        "reset": {"disabled": True},
        "normalization": {"disabled": True, "normalized_data_min": -10.0, "normalized_data_max": 10.0},
        "stagnation": {"disabled": True, "period": 500},
        "states_convergence": {"disabled": False, "start": 10, "last_obs_num": 10, "reward_tolerance": 0.5,
                               "final_eps_perc": 0.2},
    }
    return config


class StatesCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="State", config=get_config())
        self.env = None
        self.hashed_observations_buffer = []
        self.period_index = []
        self.episodes_rewards = []

    # todo IDEA: ADD the state coverage check
    def run(self, observations, environment, reward, max_reward, max_total_steps) -> None:
        """
        Does the following checks on the observations :
        (1) Checks if the observations are normalized
        (2) Checks if the states are stagnating in one episode
        (3) Checks if the observations are being repeated across episodes during exploitation phase

        Args:
            observations (Tensor): the observations returned from the step functions
            environment (gym.env): the training RL environment
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.

        Returns:

        """
        if self.check_period():
            self.check_normalized_observations(observations)
            self.update_hashed_observation_buffer(environment, observations)
            self.check_states_stagnation()
            if self.is_final_step():
                self.period_index += [len(self.hashed_observations_buffer)]
                self.episodes_rewards += [reward]
                self.check_states_converging(max_reward, max_total_steps)
                # todo CR: ask Darshan about variance

    def update_hashed_observation_buffer(self, environment, observations):
        """
        Stores the observations in the buffer

        Args:
            environment (gym.env): the training RL environment
            observations (Tensor): the observations returned from the step functions
        """
        observation_shape = environment.observation_space.shape
        if observations.shape == observation_shape:
            hashed_obs = str(hashlib.sha256(str(observations).encode()).hexdigest())
            self.hashed_observations_buffer += [hashed_obs]
        else:
            for obs in observations:
                hashed_obs = str(hashlib.sha256(str(obs).encode()).hexdigest())
                self.hashed_observations_buffer += [hashed_obs]

    def check_states_stagnation(self):
        """
        Checks whether the observations are stagnating, meaning that the environment is consistently rendering the same
        observations throughout an episode.
        """
        if self.config["stagnation"]["disabled"]:
            return
        if (len(self.hashed_observations_buffer) % self.config["stagnation"]["period"]) == 0:
            if all((obs == self.hashed_observations_buffer[-1])
                   for obs in self.hashed_observations_buffer[-self.config["stagnation"]["period"]:]):
                self.error_msg.append(self.main_msgs['observations_are_similar'].format(
                    self.config["stagnation"]["stagnated_data_nbr_check"]))

    def check_states_converging(self, max_reward, max_total_steps):
        """
        Compares the observations produced by the environment tin multiple episodes during the exploitation phase of
        the learning process andchecks whether the environment is producing the same sequence of observations. This
        check can help detect when the agent is stuck in a local optima. Note that this check is only performed when
        the average reward is far from the maximum reward threshold.
        Note that this check is only performed when the average reward is far from the maximum reward threshold.

        Args:
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
        """
        if self.config["states_convergence"]["disabled"]:
            return

        if (len(self.period_index) >= self.config["states_convergence"]["start"]) and \
                (statistics.mean(self.episodes_rewards) < max_reward * self.config["states_convergence"][
                    "reward_tolerance"]) and (self.step_num >= max_total_steps * (self.config["exploitation_perc"])):
            final_obs = []
            for i in self.period_index:
                starting_index = i - self.config["states_convergence"]["last_obs_num"]
                final_obs.append(self.hashed_observations_buffer[starting_index:i])
            if all((final_obs[i] == final_obs[i + 1]) for i in range(len(final_obs) - 1)):
                self.error_msg.append(self.main_msgs['observations_are_similar'])
            self.period_index = []
            self.episodes_rewards = []

    def check_normalized_observations(self, observations):
        """
        Checks whether the observations are normalized

        Args:
            observations (Tensor): a batch of observations
        """
        if self.config["normalization"]["disabled"]:
            return

        max_data = self.config["normalization"]["normalized_data_max"]
        min_data = self.config["normalization"]["normalized_data_min"]

        if torch.max(observations) > max_data or torch.min(observations) < min_data:
            self.error_msg.append(self.main_msgs['observations_unnormalized'])
            return

    # This function does decode the hashed observation, we may need it
    def decode_sha256(self, hashed_obs):
        bytes_obj = bytes.fromhex(hashed_obs)

        # Convert the bytes to a string
        str_obj = bytes_obj.decode()

        # Parse the string representation of the tensor
        decoded_obs = torch.tensor(eval(str_obj))
