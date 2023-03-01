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
        "reset": {"disabled": False},
        "normalization": {"disabled": True, "normalized_data_min": [-1.0], "normalized_data_max": [1.0]},
        "stagnation": {"disabled": False, "stagnated_data_nbr_check": 10},
    }
    return config


class OnTrainStatesCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainState", config=get_config())
        self.env = None
        self.check_reset = False
        self.observations_buffer = torch.Tensor([])

    def run(self, observations, steps: int, done: bool, environment) -> None:
        if self.check_period():
            self.check_reset_is_called(steps, done, environment)
            self.check_normalized_observations(observations)
            self.check_states_stagnation(observations)
            # TODO: check state stagnation per episode (periodic)
            # TODO: ask Darshan about variance

    def check_states_stagnation(self, observations):
        if self.config["stagnation"]["disabled"]:
            return
        self.observations_buffer = torch.cat([self.observations_buffer, observations])
        if (len(self.observations_buffer) % self.config["stagnation"]["stagnated_data_nbr_check"]) == 0:
            if torch.all(torch.eq(self.observations_buffer[-self.config["stagnation"]["stagnated_data_nbr_check"]:],
                                  self.observations_buffer[0])):
                self.error_msg.append(self.main_msgs['observations_are_similar'].format(
                    self.config["stagnation"]["stagnated_data_nbr_check"]))

    def check_reset_is_called(self, steps, done, environment):
        if self.config["reset"]["disabled"]:
            return
        if self.env is None:
            self.env = environment
            self.create_wrapper()

        if self.check_reset and (not self.env.reset.called):
            self.error_msg.append(self.main_msgs['reset_was_not_called'])

        # TODO: make done, steps .... observed automatically in the interface
        if done or steps > environment.spec.max_episode_steps:
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

    def track_func(self, func):
        def wrapper(*args, **kwargs):
            wrapper.called = True
            return func(*args, **kwargs)

        wrapper.called = False
        return wrapper

    def func_called(self):
        return self.env.reset.called

    def reset_called(self):
        self.env.reset.called = False

    def create_wrapper(self):
        self.env.reset = self.track_func(self.env.reset)
