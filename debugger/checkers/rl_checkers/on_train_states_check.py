import torch
from debugger.debugger_interface import DebuggerInterface


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "period": 1,
        "reset": {"disabled": False}
    }
    return config


class OnTrainStatesCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="OnTrainState", config=get_config())
        self.env = None
        self.check_reset = False

    def run(self, steps: int, done: bool, environment) -> None:
        if self.check_period():
            self.check_reset_is_called(steps, done, environment)
            # TODO: check state stagnation overall
            # TODO: check state stagnation per episode (periodic)
            # TODO: ask Darshan about variance

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
