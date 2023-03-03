from debugger import DebuggerInterface
import debugger as debugger_lib
from debugger.utils import settings
from debugger.utils.registry import registry
import inspect
import yaml
import copy


class DebuggerFactory:
    def __init__(self):
        self.logger = settings.set_logger()
        self.debuggers = dict()
        self.params = {}
        self.params_iters = dict()
        self.step_num = 0

    def set_debugger(self, config):
        """
        Set the `debugger` object with the provided `config`.

        Args:
            config (dict): The dictionary of the checks names to be done.
        """

        self.init_params_iteration(config)
        if config["debugger"]["kwargs"]["check_type"]:
            config = config["debugger"]["kwargs"]["check_type"]
            for debugger_config in config:
                debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
                debugger = debugger_fn()
                self.debuggers[debugger_config["name"]] = debugger

    def init_params_iteration(self, config):
        """
        Set the `params_iters` attribute with the provided `config`.
        """

        params = config["debugger"]["kwargs"]["params"]
        self.params_iters = {key: 0 for key, val in params["variable"]}
        self.params_iters.update({key: -1 for key, val in params["constant"]})

    # def create_env_wrapper(self, environment):
    #     self.create_wrapper(environment)

    # if max_steps_per_episode is not None:
    #     for checkers in self.debuggers.values():
    #         if checkers.max_steps_per_episode == max_steps_per_episode:
    #             break
    #         checkers.max_steps_per_episode = max_steps_per_episode

    def track_func(self, func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            self.params['observations'], self.params['reward'], self.params['done'], _ = results
            self.params_iters['observations'] += 1
            self.params_iters['reward'] += 1
            self.params_iters['done'] += 1
            self.step_num += 1
            return results

        return wrapper

    def create_env_wrapper(self, kwargs):
        if "environment" in kwargs.keys():
            kwargs["environment"].step = self.track_func(kwargs["environment"].step)

    def is_final_step_of_ep(self):
        if self.params_iters['done'] or (self.step_num >= self.params_iters['max_steps_per_episode']):
            return True
        return False

    def set_parameters(self, **kwargs):
        """
        Set the `params` dictionary and the `params_iters` dictionary with the provided `kwargs`.
        """

        self.create_env_wrapper(kwargs)

        for key, value in kwargs.items():
            self.params[key] = copy.deepcopy(value)
            if self.params_iters[key] != -1:
                self.params_iters[key] += 1

    def react(self, messages, fail_on=False):
        """
        Reacts to the provided `messages` by either raising an exception or logging a warning, depending on the value of
         `fail_on`.

        Args:
            messages (list): list of error messages to be displayed
            fail_on (bool): if True it raises an exception otherwise it only displays the error
        """
        if len(messages) > 0:
            for message in messages:
                if fail_on:
                    self.logger.error(message)
                    raise Exception(message)
                else:
                    self.logger.warning(message)

    def run(self):
        """
        Runs the `debugger` objects in the `debuggers` dictionary.
        """
        for debugger in self.debuggers.values():
            arg_names = inspect.getfullargspec(debugger.run).args[1:]
            params_iters = [self.params_iters[key] for key in arg_names]
            if all(
                    param_iter == -1 or param_iter >= debugger.iter_num + 1
                    for param_iter, arg_name in zip(params_iters, arg_names)
            ):
                debugger.increment_iteration()
                kwargs = {arg: self.params[arg] for arg in arg_names}
                debugger.run(**kwargs)
                self.react(debugger.error_msg)
                debugger.reset_error_msg()

    def run_debugging(self, **kwargs):
        """
        Calls the `set_parameters` method with the provided `kwargs`, and then calls the `run` method, to start running
        the checks
        """
        try:
            self.set_parameters(**kwargs)
            self.run()
        except Exception as e:
            self.react(messages=[f"Error: {e}"], fail_on=True)
            # Attempt to recover from the error and continue
            pass

    def set_config(self, config=None, config_path=None):
        """
        Set the `debugger` object with the provided `config` or `config_path`.

        Args:
            config (dict): the configuration dict
            config_path (str): The path to the configuration dict
        """

        if config is not None:
            self.set_debugger(config)
        if config_path is not None:
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)
                self.set_debugger(loaded_config)

    @staticmethod
    def register(checker_name: str, checker_class: DebuggerInterface) -> None:
        registry.register(checker_name, checker_class, checker_class)
