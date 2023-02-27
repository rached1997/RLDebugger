import torch

from debugger.debugger_interface import DebuggerInterface
import gym


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "start": 100,
        "Period": 1,
        "observations_var_coef_thresh": 0.001,
        "reset_func_check": {
            "disabled": True,
        }
    }
    return config


class OnTrainStatesCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainState", config=get_config())
        self.env = None
        self.check_reset = False
        self.max_steps = None

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

    def run(self, steps: int, done: bool, environment: gym.envs, max_steps: int) -> None:
        if self.env is None:
            self.env = environment
            self.create_wrapper()

        if self.max_steps is None:
            self.max_steps = max_steps

        if self.check_reset and (not self.env.reset.called):
            self.error_msg.append(self.main_msgs['reset_was_not_called'])

        if done or steps > self.max_steps:
            self.check_reset = True
