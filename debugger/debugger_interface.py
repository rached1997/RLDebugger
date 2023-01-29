from hive.utils.registry import Registrable
from hive.debugger.utils import settings


class DebuggerInterface(Registrable):
    def __init__(self, check_type, config):
        self.main_msgs = settings.load_messages()
        self.config = config
        self.check_type = check_type
        self.period = config["Period"]

    @classmethod
    def type_name(cls):
        return "debugger"