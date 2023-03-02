import math

import torch
import math
from debugger.utils.registry import Registrable
from debugger.utils import settings

# todo add the checks from the blog
# todo cover the testing during the training
class DebuggerInterface(Registrable):
    def __init__(self, check_type, config):
        self.main_msgs = settings.load_messages()
        self.config = config
        self.check_type = check_type
        self.period = config["Period"]
        self.iter_num = 0
        self.error_msg = list()
        self.done = False
        self.step_num = -1
        self.max_steps_per_episode = math.inf

    def check_period(self):
        """
        Checks if the period of the check has been reached

        Returns:
            True if the period is reached. False otherwise.
        """
        return ((self.period != 0) and (self.iter_num % self.period == 0)) or (
                (self.period == 0) and (self.iter_num == 1))

    def increment_iteration(self):
        """
            Increments the iteration
        """
        self.iter_num += 1

    def reset_error_msg(self):
        """
            empties the error messageslist
        """
        self.error_msg = list()

    @classmethod
    def type_name(cls):
        return "debugger"

    def track_func(self, func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            self.done = results[2]
            self.step_num += 1
            return results
        return wrapper

    def create_wrapper(self, environment):
        environment.step = self.track_func(environment.step)
        self.step_num = 0

    def is_final_step_of_ep(self):
        if self.done or (self.step_num >= self.max_steps_per_episode):
            return True
        return False

    # todo add this to all checkers
    def flush(self, var_list_name=None, var_list_obj=None):
        if var_list_name is not None:
            for var_name in var_list_name:
                if getattr(self, var_name, False):
                    setattr(self, var_name, None)
        if var_list_obj is not None:
            for i in range(len(var_list_obj)):
                var_list_obj[i] = None

    def flush_all(self):
        for name, value in vars(self).items():
            vars(self)[name] = None

    def flush_all_tensors(self):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                vars(self)[name] = None
