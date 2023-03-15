from debugger import DebuggerInterface


def get_config():
    config = {"Period": 100}
    return config


class CustomChecker(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="CustomChecker", config=get_config())
        # you can add other attributes

    #  You can define other functions

    def run(self, observed_param):
        if self.check_period():
            # Do some instructions ....
            self.error_msg.append(
                "The custom checker is integrated successfully!;"
                "The value of the observed_param is {} ".format(observed_param)
            )
