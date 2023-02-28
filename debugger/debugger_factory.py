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

    def set_parameters(self, **kwargs):
        """
        Set the `params` dictionary and the `params_iters` dictionary with the provided `kwargs`.
        """
        for key, value in kwargs.items():
            # self.params[key] = copy.deepcopy(value)
            self.params[key] = value
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
            if not (defaults is None):
                defaults_dict = dict(zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
            params_iters = [self.params_iters[key] for key in arg_names]
            # TODO: review this
            if all(((params_iters[i] == -1)
                    or (params_iters[i] >= (debugger.iter_num + 1))
                    or (arg_names[i] in defaults_dict.keys()))
                   for i in range(len(params_iters))):
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
        # todo we can add try except to avoid crushing the training if there is any error
        self.set_parameters(**kwargs)
        self.run()

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
