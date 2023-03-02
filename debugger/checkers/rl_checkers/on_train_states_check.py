import hashlib
import statistics

import torch
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import almost_equal



def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 1,
        "reset": {"disabled": True},
        "normalization": {"disabled": True, "normalized_data_min": [-1.0], "normalized_data_max": [1.0]},
        "stagnation": {"disabled": True, "stagnated_data_nbr_check": 10},
        "states_convergence": {"disabled": False, "num_eps_to_check": 10, "last_obs_perc": 0.1, "reward_tolerance": 0.5, "final_eps_perc": 0.2},
    }
    return config


class OnTrainStatesCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainState", config=get_config())
        self.env = None
        self.check_reset = False
        self.observations_buffer = []
        self.period_index = []
        self.episodes_rewards = []

    def run(self, observations, environment, reward, max_reward, max_total_steps) -> None:
        if self.check_period():
            self.check_reset_is_called(environment)
            self.check_normalized_observations(observations)

            for obs in observations:
                hashed_obs = str(hashlib.sha256(str(obs).encode()).hexdigest())
                self.observations_buffer += [hashed_obs]

            if self.is_final_step_of_ep():
                self.period_index += [len(self.observations_buffer) - 1]
                self.episodes_rewards += [reward]

                self.check_states_stagnation()
                self.check_states_converging(max_reward, max_total_steps)
                # TODO: ask Darshan about variance

    def check_states_stagnation(self, ):
        if self.config["stagnation"]["disabled"]:
            return
        if (len(self.observations_buffer) % self.config["stagnation"]["stagnated_data_nbr_check"]) == 0:
            if all((obs == self.observations_buffer[0])
                   for obs in self.observations_buffer[-self.config["stagnation"]["stagnated_data_nbr_check"]:]):
                self.error_msg.append(self.main_msgs['observations_are_similar'].format(
                    self.config["stagnation"]["stagnated_data_nbr_check"]))

    # todo check again if the implementation is correct
    def check_states_converging(self, max_reward, max_total_steps):
        if self.config["states_convergence"]["disabled"]:
            return

        if (len(self.period_index) >= self.config["states_convergence"]["num_eps_to_check"]) and \
                (statistics.mean(self.episodes_rewards) < max_reward * self.config["states_convergence"]["reward_tolerance"]) and \
                (self.step_num >= max_total_steps * (1 - self.config["states_convergence"]["final_eps_perc"])):
            previous_ep_index = 0
            final_obs = []
            for i in self.period_index:
                # (i-previous_ep_index) measures the length of the ep
                starting_index = i - int((i - previous_ep_index) * self.config["states_convergence"]["last_obs_perc"])
                final_obs += self.observations_buffer[starting_index:i + 1]
            if all((obs == final_obs[0]) for obs in final_obs):
                self.error_msg.append(self.main_msgs['observations_are_similar'])
        self.period_index = []
        self.episodes_rewards = []

    # todo this function doesn't work with deep copy maybe change it
    def check_reset_is_called(self, environment):
        if self.config["reset"]["disabled"]:
            return
        if self.env is None:
            self.env = environment
            self.create_rest_wrapper()

        if self.check_reset and (not self.env.reset.called):
            self.error_msg.append(self.main_msgs['reset_was_not_called'])

        if self.is_final_step_of_ep():
            self.check_reset = True

    def check_normalized_observations(self, observations):
        #  todo this check is not correct, verify it with Darashan (example in the cartpool some values are > 1 )
        if self.config["normalization"]["disabled"]:
            return

        mas = torch.max(observations)
        mis = torch.min(observations)
        avgs = torch.mean(observations * 1.0)
        stds = torch.std(observations * 1.0)

        if any([(mas > data_max) for data_max in self.config["normalization"]["normalized_data_max"]]) and \
                any([(mis < data_min) for data_min in self.config["normalization"]["normalized_data_min"]]):
            return
        elif not (almost_equal(stds, 1.0) and almost_equal(avgs, 0.0)):
            self.error_msg.append(self.main_msgs['observations_unnormalized'])

    def track_reset_func(self, func):
        def wrapper(*args, **kwargs):
            wrapper.called = True
            return func(*args, **kwargs)

        wrapper.called = False
        return wrapper

    def func_called(self):
        return self.env.reset.called

    def reset_called(self):
        self.env.reset.called = False

    def create_rest_wrapper(self):
        self.env.reset = self.track_reset_func(self.env.reset)

    # This function does decode the hashed observation, we may need it
    def decode_sha256(self, hashed_obs):
        bytes_obj = bytes.fromhex(hashed_obs)

        # Convert the bytes to a string
        str_obj = bytes_obj.decode()

        # Parse the string representation of the tensor
        decoded_obs = torch.tensor(eval(str_obj))
