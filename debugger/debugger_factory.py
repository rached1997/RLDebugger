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

        if config is not None:
            self.set_debugger(config)

    def set_debugger(self, config):
        config = config["debugger"]["kwargs"]["check_type"]
        for debugger_config in config:
            debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
            debugger = debugger_fn()
            self.debuggers[debugger_config["name"]] = debugger

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            # TODO: change deepcopy
            self.params[key] = copy.deepcopy(value)

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
            debugger.increment_iteration()
            args = inspect.getfullargspec(debugger.run).args[1:]
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

