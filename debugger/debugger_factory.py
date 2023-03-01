from pathlib import Path
import debugger
import debugger as debugger_lib
from debugger.utils import settings
from debugger.utils.registry import registry
import inspect
import yaml
import copy


class DebuggerFactory:
    def __init__(self, config=None, app_path=None):
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, "logger")
        self.logger = settings.file_logger(log_fpath, "logger")
        self.debuggers = dict()
        self.params = {}
        self.params_iters = dict()
        if config is not None:
            self.set_debugger(config)

    def set_debugger(self, config):
        """
        Set the `debugger` object with the provided `config`.

        Args:
            config (dict): The dictionary of the checks names to be done.
        """

        self.set_params_iteration(config)
        if config["debugger"]["kwargs"]["check_type"]:
            config = config["debugger"]["kwargs"]["check_type"]
            for debugger_config in config:
                debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
                debugger = debugger_fn()
                self.debuggers[debugger_config["name"]] = debugger

    def set_params_iteration(self, config):
        """
        Set the `params_iters` attribute with the provided `config`.
        """

        params = config["debugger"]["kwargs"]["params"]
        self.params_iters = {key: -1 if val == "constant" else 0 for key, val in params.items()}

    def create_env_wrapper(self, environment=None, max_steps_per_episode=None):
        if environment is not None:
            for checkers in self.debuggers.values():
                if checkers.step_num != -1:
                    break
                checkers.create_wrapper(environment)

        if max_steps_per_episode is not None:
            for checkers in self.debuggers.values():
                if checkers.max_steps_per_episode == max_steps_per_episode:
                    break
                checkers.max_steps_per_episode = max_steps_per_episode

    def set_parameters(self, **kwargs):
        """
        Set the `params` dictionary and the `params_iters` dictionary with the provided `kwargs`.
        """
        if "environment" in kwargs.keys():
            self.create_env_wrapper(environment=kwargs["environment"])
        if "max_steps_per_episode" in kwargs.keys():
            self.create_env_wrapper(max_steps_per_episode=kwargs["max_steps_per_episode"])

        for key, value in kwargs.items():
            self.params[key] = copy.deepcopy(value)
            # self.params[key] = value
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
            argspec = inspect.getfullargspec(debugger.run)
            arg_names = argspec.args[1:]
            defaults = argspec.defaults
            defaults_dict = {}
            if defaults:
                defaults_dict = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
            params_iters = [self.params_iters[key] for key in arg_names]
            if all(
                    param_iter == -1 or param_iter >= debugger.iter_num + 1 or arg_name in defaults_dict.keys()
                    for param_iter, arg_name in zip(params_iters, arg_names)
            ):
                debugger.increment_iteration()
                kwargs = {arg: self.params[arg] if arg in self.params.keys() else defaults_dict[arg]
                          for arg in arg_names}
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
    def register(checker_name: str, checker_class: debugger.DebuggerInterface) -> None:
        registry.register(checker_name, checker_class, checker_class)
