from pathlib import Path
import debugger as debugger_lib
from debugger.utils import settings
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
        self.set_params_iteration(config)
        config = config["debugger"]["kwargs"]["check_type"]
        for debugger_config in config:
            debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
            debugger = debugger_fn()
            self.debuggers[debugger_config["name"]] = debugger

    def set_params_iteration(self, config):
        self.params_iters = config["debugger"]["kwargs"]["params"]

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = copy.deepcopy(value)
            if self.params_iters[key] != -1:
                self.params_iters[key] += 1

    def react(self, messages, fail_on=False):
        if len(messages) > 0:
            for message in messages:
                if fail_on:
                    self.logger.error(message)
                    raise Exception(message)
                else:
                    self.logger.warning(message)

    def run(self):
        for debugger in self.debuggers.values():
            args = inspect.getfullargspec(debugger.run).args[1:]
            params_iters = [self.params_iters[key] for key in args]
            if all(((val == -1) or (val == (debugger.iter_num + 1))) for val in params_iters):
                debugger.increment_iteration()
                kwargs = {arg: self.params[arg] for arg in args}
                debugger.run(**kwargs)
                self.react(debugger.error_msg)
                debugger.reset_error_msg()

    def run_debugging(self, **kwargs):
        self.set_parameters(**kwargs)
        self.run()

    def set_config(self, config=None, config_path=None):
        if config is not None:
            self.set_debugger(config)
        if config_path is not None:
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)
                self.set_debugger(loaded_config)
