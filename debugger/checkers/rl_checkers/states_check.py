import hashlib
import statistics

import torch

from debugger.config_data_classes.rl_checkers.states_config import StatesConfig
from debugger.debugger_interface import DebuggerInterface


class StatesCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="State", config=StatesConfig)
        self.env = None
        self.hashed_observations_buffer = []
        self.period_index = []
        self.episodes_rewards = []

    # todo IDEA: ADD the state coverage check
    def run(
        self, observations, environment, reward, max_reward, max_total_steps
    ) -> None:
        """
        The states represents the main output of the environment that the agent uses to predict the next action to
        take. Generally DRL is applied in complex environments that have either a continuous observations space or a
        very high number of possible states. One of the wrong behaviours of the agent that can be detected through
        the states is that the agent is repeating it's behaviour without being close to reach its expected goal. The
        repeated behaviour can be within one episode or the agent is repeating a sequence of actions in each episode
        which can be due to issues such as the agent being stuck in a local optimum. For example, an agent is
        learning to solve a maze, however the agent didn't do enough exploration, and is repeating its beahviour in
        each episode, which means that every episode the agent follows the same path which can't lead it to reach its
        goal. This wrong bihaviour can be detecetd by analysung the behaviour of the observations returned in each
        episode.


        The states check does the following checks on the observations :
            (1) Checks if the observations are normalized,
            (2) Checks if the states are stagnating in one episode
            (3) Checks if the observations are being repeated across episodes during exploitation phase

        The potential root causes behind the warnings that can be detected are
            - Missing exploration or suboptimal exploration rate (checks triggered : 2,3)
            - Bad conception of the step function (checks triggered : 1, 2)
            - Unstable learning (checks triggered : 2)
            - Coding error (e.g. saving the same data in the memory) (checks triggered : 1, 2, 3)

        The recommended fixes for the detected issues :
            - Add more exploration (checks that can be fixed: 2,3)
            - Change the ratio of exploration exploitation (checks that can be fixed: 2,3)
            - Check if the observations are normalized (checks that can be fixed: 1)
            - Check if the step function is working correctly (checks that can be fixed: 1,2,3 )
            - verify that if the agent is learning correctly, and change its architecture (checks that can be fixed: 2, 3 )

        Args:
            observations (Tensor): the observations returned from the step functions
            environment (gym.env): the training RL environment
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.

        Returns:

        """
        if self.is_final_step():
            self.period_index += [len(self.hashed_observations_buffer)]
            self.episodes_rewards += [reward]
        if self.skip_run(self.config.skip_run_threshold):
            return
        if self.check_period():
            self.check_normalized_observations(observations)
            self.update_hashed_observation_buffer(environment, observations)
            self.check_states_stagnation()
            self.check_states_converging(max_reward, max_total_steps)

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
        if self.config.stagnation.disabled:
            return
        if (len(self.hashed_observations_buffer) % self.config.stagnation.period) == 0:
            if all(
                (obs == self.hashed_observations_buffer[-1])
                for obs in self.hashed_observations_buffer[
                    -self.config.stagnation.period :
                ]
            ):
                self.error_msg.append(
                    self.main_msgs["observations_are_similar"].format(
                        self.config.stagnation.stagnated_data_nbr_check
                    )
                )

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
        if self.config.states_convergence.disabled:
            return
        if len(self.period_index) >= self.config.states_convergence.start:
            if (
                statistics.mean(self.episodes_rewards)
                < max_reward * self.config.states_convergence.reward_tolerance
            ) and (self.step_num >= max_total_steps * (self.config.exploitation_perc)):
                final_obs = []
                for i in self.period_index:
                    starting_index = i - self.config.states_convergence.last_obs_num
                    final_obs.append(self.hashed_observations_buffer[starting_index:i])
                if all(
                    (final_obs[i] == final_obs[i + 1])
                    for i in range(len(final_obs) - 1)
                ):
                    self.error_msg.append(self.main_msgs["observations_are_similar"])
                self.period_index = []
                self.episodes_rewards = []

    def check_normalized_observations(self, observations):
        """
        Checks whether the observations are normalized

        Args:
            observations (Tensor): a batch of observations
        """
        if self.config.normalization.disabled:
            return

        max_data = self.config.normalization.normalized_data_max
        min_data = self.config.normalization.normalized_data_min

        if torch.max(observations) > max_data or torch.min(observations) < min_data:
            self.error_msg.append(self.main_msgs["observations_unnormalized"])
            return

    # This function does decode the hashed observation, we may need it
    def decode_sha256(self, hashed_obs):
        bytes_obj = bytes.fromhex(hashed_obs)

        # Convert the bytes to a string
        str_obj = bytes_obj.decode()

        # Parse the string representation of the tensor
        decoded_obs = torch.tensor(eval(str_obj))
