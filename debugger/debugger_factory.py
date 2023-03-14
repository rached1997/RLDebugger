from debugger import DebuggerInterface
import debugger as debugger_lib
from debugger.utils import settings
from debugger.utils.registry import registry
import yaml

from debugger.utils.settings import react, load_default_config


class DebuggerFactory:
    def __init__(self):
        self.logger = settings.set_logger()
        self.wandb_logger = settings.set_wandb_logger()
        self.debuggers = dict()
        self.params = {}
        self.params_iters = dict()
        self.step_num = -1
        self.training = True

    def track_func(self, func_step, func_reset):
        """
        Create a wrapper for the step and reset functions of the environment that allows tracking the values of their
        outputs, including the observation, the next observation, the reward, done, and the step number.

        Args:
            func_step: the step function of the RL environment.
            func_reset: the reset function of the RL environment.

        Returns: The wrapped step and reset functions of the environment

        """

        def step_wrapper(*args, **kwargs):
            """
            Create a wrapper function for the step function of the environment.

            Returns:A wrapped version of the step function that returns the same result as the original step function.
            """
            if self.step_num == 0 or self.is_final_step_of_ep():
                self.params['next_observations'] = self.params['observations']
                self.params['reward'] = 0
            results = func_step(*args, **kwargs)
            if self.training:
                self.params['observations'] = self.params['next_observations']
                self.params['next_observations'] = results[0]
                self.params['reward'] += results[1]
                self.params['done'] = results[2]
                self.params_iters['observations'] += 1
                self.params_iters['next_observations'] += 1
                self.params_iters['reward'] += 1
                self.params_iters['done'] += 1
                self.step_num += 1
            return results

        def reset_wrapper(*args, **kwargs):
            """
            Create a wrapper function for the reset function of the environment.

            Returns: A wrapped version of the reset function that returns the same result as the original reset function
            """
            results = func_reset(*args, **kwargs)
            if self.training:
                self.params['observations'] = results
                self.params_iters['observations'] += 1
            return results

        self.params['observations'] = None
        self.params['reward'] = 0
        self.params['done'] = False
        self.step_num = 0
        return step_wrapper, reset_wrapper

    def create_wrappers(self, kwargs):
        """
        creates a wrapper for  the step and reset functions of the RL environment in order to track the observation,
        the next observation, the reward, done, and the step number.

        Args:
            kwargs: the same kwargs received by the function set_parameters
        """
        if "environment" in kwargs.keys() and self.step_num == -1:
            func_wrappers = self.track_func(kwargs["environment"].step, kwargs["environment"].reset)
            kwargs["environment"].step, kwargs["environment"].reset = func_wrappers

    def is_final_step_of_ep(self):
        """
        Check if the current step represents the last step of an episode by checking if 'done' is True or if the step
        number has reached the maximum value.

        Returns (bool): returns True if the step is the last one in an episode, and False otherwise.
        """
        if self.params['done'] or (
                (self.step_num > 0) and ((self.step_num % self.params['max_steps_per_episode']) == 0)):
            return True
        return False

    def set_parameters(self, **kwargs):
        """
        Set the `params` dictionary and the `params_iters` dictionary with the provided `kwargs`.
        """
        for key, value in kwargs.items():
            # self.params[key] = copy.deepcopy(value)
            self.params[key] = value
            if self.params_iters[key] != -1:
                self.params_iters[key] += 1
        self.create_wrappers(kwargs)

    def run(self):
        """
        Runs the `debugger` objects in the `debuggers` dictionary.
        """
        for debugger in self.debuggers.values():
            arg_names = debugger.get_arg_names()
            kwargs = {}
            is_ready = True
            for arg in arg_names:
                param_iter = self.params_iters[arg]
                if not (param_iter == -1 or param_iter >= debugger.iter_num + 1):
                    is_ready = False
                    break
                else:
                    kwargs[arg] = self.params[arg]
            if is_ready:
                debugger.step_num = self.step_num
                if debugger.max_total_steps is None:
                    debugger.max_total_steps = self.params["max_total_steps"]
                debugger.increment_iteration()
                debugger.run(**kwargs)
                if debugger.wandb_metrics:
                    self.wandb_logger.plot(debugger.wandb_metrics)
                react(self.logger, debugger.error_msg)
                debugger.reset_error_msg()

    def run_debugging(self, **kwargs):
        """
        Calls the `set_parameters` method with the provided `kwargs`, and then calls the `run` method, to start running
        the checks
        """
        try:
            # print(self.step_num)
            if self.training:
                self.set_parameters(**kwargs)
                if "max_total_steps" in self.params.keys():
                    self.run()
                else:
                    react(logger=self.logger,
                          messages=[f"Warning: Please provide value for max_steps_per_episode to run the debugger"],
                          fail_on=False)
                # self.wandb_logger.log_scalar("step_num", self.step_num, "debugger")
        except Exception as e:
            # TODO: put it back to false and make it run once
            react(logger=self.logger, messages=[f"Error: {e}"], fail_on=True)
            # Attempt to recover from the error and continue
            pass

    def set_config(self, config_path=None):
        """
        Set the `debugger` object with the provided `config_path` or with the default config.

        Args:
            config (dict): the configuration dict
            config_path (str): The path to the configuration dict
        """
        if config_path is None:
            config_path = load_default_config()

        with open(config_path) as f:
            config = yaml.safe_load(f)
            # init params
            params = config["debugger"]["kwargs"]["params"]
            self.params_iters.update({key: 0 for key in params["variable"]})
            # the constant values should be initialized with -2 which indicates that they are constants but still
            # doesn't have a value yet.
            self.params_iters.update({key: -2 for key in params["constant"]})

            if config["debugger"]["kwargs"]["check_type"]:
                config = config["debugger"]["kwargs"]["check_type"]
                for debugger_config in config:
                    debugger_fn, _ = debugger_lib.get_debugger(debugger_config, debugger_config["name"])
                    debugger = debugger_fn()
                    if "period" in debugger_config.keys():
                        debugger.period = debugger_config["period"]
                    self.debuggers[debugger_config["name"]] = debugger

                # set internal parameters of the debuggers
                for debugger in self.debuggers.values():
                    debugger.set_params(self.is_final_step_of_ep)

    @staticmethod
    def register(checker_name: str, checker_class: DebuggerInterface) -> None:
        """
        Register a new checker. This is mostly useful for integrating a custom checker.

        Args:
            checker_name (str): The name of the checker.
            checker_class (DebuggerInterface): The class of the debugger.
        """
        registry.register(checker_name, checker_class, checker_class)

    def turn_off(self):
        """
        Turn off the debugger. This function is useful when there are testing episodes that occur during training.
        """
        self.training = False

    def turn_on(self):
        """
        Turn on the debugger. This function is useful when there are testing episodes that occur during training.
        """
        self.training = True

    def set_custom_wand_logger(self, project, name, dir=None, mode=None, id=None, resume=None, start_method=None,
                               **kwargs):
        self.wandb_logger.custom_wandb_logger(project, name, dir, mode, id, resume, start_method, **kwargs)
