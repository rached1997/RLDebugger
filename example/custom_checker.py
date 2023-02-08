from debugger import DebuggerInterface


def get_config():
    config = {"Period": 0, "threshold": 0.05}
    return config


class CustomChecker(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="CustomChecker", config=get_config())
        # you can add other attributes

    #  You can define other functions

    def run(self, observed_param):
        if self.check_period():
            self.error_msg.append(
                "your custom checker is running successfully!;"
                " your observed parameter is {} and your threshold is {}".format(observed_param,
                                                                                 self.config["threshold"]))



