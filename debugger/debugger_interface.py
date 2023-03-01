from debugger.utils.registry import Registrable
from debugger.utils import settings


class DebuggerInterface(Registrable):
    def __init__(self, check_type, config):
        self.main_msgs = settings.load_messages()
        self.config = config
        self.check_type = check_type
        self.period = config["Period"]
        self.iter_num = 0
        self.error_msg = list()

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

    # TODO: add flush function
