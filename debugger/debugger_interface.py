import inspect
import torch
from debugger.utils.registry import Registrable
from debugger.utils import settings
from debugger.utils.utils import get_device


class DebuggerInterface(Registrable):
    def __init__(self, check_type, config):
        self.main_msgs = settings.load_messages()
        self.config = config
        self.check_type = check_type
        self.period = config.period
        self.iter_num = 0
        self.error_msg = list()
        self.step_num = None
        self.skipped_step_num = 0
        self.old_step_num = -1
        self.is_final_step = None
        self.max_total_steps = None
        self.arg_names = None
        self.wandb_metrics = {}
        self.device = get_device()

    def check_period(self):
        """
        Checks if the period of the check has been reached

        Returns:
            True if the period is reached. False otherwise.
        """
        return ((self.period != 0) and (self.iter_num % self.period == 0)) or (
            (self.period == 0) and (self.iter_num == 1)
        )

    def skip_run(self, threshold):
        step_diff = self.step_num - self.old_step_num
        if self.old_step_num == -1 or step_diff >= threshold:
            self.old_step_num = self.step_num
            return False
        self.iter_num -= 1
        self.skipped_step_num += 1
        return True

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

    def set_params(self, is_final_step):
        self.is_final_step = is_final_step

    def get_arg_names(self):
        if self.arg_names is None:
            self.arg_names = inspect.getfullargspec(self.run).args[1:]
        return self.arg_names

    def get_number_off_calls(self):
        return self.iter_num + self.skipped_step_num + 1
