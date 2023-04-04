from debugger import DebuggerInterface
import debugger as debugger_lib
from debugger.utils import settings
from debugger.utils.registry import registry
import yaml
import torch
from debugger.utils.settings import react, load_default_config


class DebuggerFactory:
    def __init__(self):
        self.logger = settings.set_logger()
        self.wandb_logger = None
        self.debuggers = dict()
        self.observed_params = {}
        self.observed_params_update_nums = dict()
        self.step_num = -1
        self.training = True
        self.device = "cpu"
        self.display_ratio_period = 0.05

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
                self.observed_params["next_observations"] = self.observed_params[
                    "observations"
                ]
                self.observed_params["reward"] = 0
            results = func_step(*args, **kwargs)
            if self.training:
                self.observed_params["observations"] = self.observed_params[
                    "next_observations"
                ]
                self.observed_params["next_observations"] = torch.tensor(
                    results[0], device=self.get_device()
                )
                self.observed_params["reward"] += results[1]
                self.observed_params["done"] = results[2]
                self.observed_params_update_nums["observations"] += 1
                self.observed_params_update_nums["next_observations"] += 1
                self.observed_params_update_nums["reward"] += 1
                self.observed_params_update_nums["done"] += 1
                self.step_num += 1
            return results

        def reset_wrapper(*args, **kwargs):
            """
            Create a wrapper function for the reset function of the environment.

            Returns: A wrapped version of the reset function that returns the same result as the original reset function
            """
            results = func_reset(*args, **kwargs)
            if self.training:
                self.observed_params["observations"] = torch.tensor(
                    results, device=self.get_device()
                )
                self.observed_params_update_nums["observations"] += 1
            return results

        self.observed_params["observations"] = None
        self.observed_params["reward"] = 0
        self.observed_params["done"] = False
        self.step_num = 0
        return step_wrapper, reset_wrapper

    def wrap_env(self, env):
        """
        creates a wrapper for  the step and reset functions of the RL environment in order to track the observation,
        the next observation, the reward, done, and the step number.

        Args:
            env: gym environment
        """
        func_wrappers = self.track_func(env.step, env.reset)
        env.step, env.reset = func_wrappers

    def is_final_step_of_ep(self):
        """
        Check if the current step represents the last step of an episode by checking if 'done' is True or if the step
        number has reached the maximum value.

        Returns (bool): returns True if the step is the last one in an episode, and False otherwise.
        """
        if self.observed_params["done"] or (
                (self.step_num > 0)
                and ((self.step_num % self.observed_params["max_steps_per_episode"]) == 0)
        ):
            return True
        return False

    def set_observed_parameters(self, **kwargs):
        """
        Set the `observed_params` dictionary and the `observed_params_update_nums` dictionary with the provided `kwargs`.
        """
        for key, value in kwargs.items():
            # self.observed_params[key] = copy.deepcopy(value)
            self.observed_params[key] = value
            if self.observed_params_update_nums[key] != -1:
                self.observed_params_update_nums[key] += 1
        if "environment" in kwargs.keys() and self.step_num == -1:
            self.wrap_env(kwargs["environment"])

    def run(self):
        """
        Runs the `debugger` objects in the `debuggers` dictionary.
        """
        for debugger in self.debuggers.values():
            arg_names = debugger.get_arg_names()
            kwargs = {}
            is_ready = True
            for arg in arg_names:
                param_update_num = self.observed_params_update_nums[arg]
                if not (
                        param_update_num == -1
                        or param_update_num >= debugger.get_number_off_calls()
                ):
                    is_ready = False
                    break
                else:
                    kwargs[arg] = self.observed_params[arg]
            if is_ready:
                debugger.step_num = self.step_num
                if debugger.max_total_steps is None:
                    debugger.max_total_steps = self.observed_params["max_total_steps"]
                debugger.increment_iteration()
                debugger.run(**kwargs)
                self.plot_wandb(debugger)
            if (self.step_num % (self.observed_params["max_total_steps"] * self.display_ratio_period)) == 0:
                react(self.logger, debugger.error_msg)
                debugger.reset_error_msg()

    def debug(self, **kwargs):
        """
        This function is used to initiate debugging of the DRL. It requires the parameters that the debugger is
        tracking in order to automatically run the checkers that are relevant to the received parameters. The
        checkers will not work unless you call the debug function and provide them with the required parameters. It
        is important to ensure that the parameter names you use in your code match those in the configuration file.
        This function can be called from different parts of the user's code. It stores the values of the parameters
        and keeps checking if there are checkers that can be run at any given moment.
        The class operates by calling the `set_parameters` method with the provided `kwargs`, and then calling the
        `run` method to start running the possible checks.

        Note : -It's very important to send the parameter environment before starting any training because it's used
        to track multiple parameters automatically during the training (e.g. rewards, observations, done ...).
        Without the environment variable being sent at the beginning the checker won't work correctly.
        - It's also recommended to send all the constant variables before starting the training
        - Make sure to use the correct names of variables in the kwargs. The names should be the same as the ones
        mentioned in the default_debugger.config file
        """
        try:
            # print(self.step_num)
            if self.training:
                self.set_observed_parameters(**kwargs)
                if "max_total_steps" in self.observed_params.keys():
                    self.run()
                else:
                    react(
                        logger=self.logger,
                        messages=[
                            f"Warning: Please provide value for max_steps_per_episode to run the debugger"
                        ],
                        fail_on=False,
                    )
                # self.wandb_logger.log_scalar("step_num", self.step_num, "debugger")
        except Exception as e:
            react(logger=self.logger, messages=[f"Error: {e}"], fail_on=True)
            # Attempt to recover from the error and continue
            pass

    def set_config(self, config_path=None):
        """
        This function is in responsible for setting the debugger's configuration. By configuration, we mean the
        parameters that the debugger will observe and the checks that will be executed.

        When you import the debugger, this function sets the default settings (see utils/config/default debugger.yml
        for more information) which provides the names of the parameters needed to execute the checks.

        In addition to the default call for this function, the user must supply his config file location where he
        mentions which checks to activate and for what period (the number of times the check will run during the
        training). The user can also specify a list of custom variables to track, which is particularly useful when
        creating his own checker. The variables provided by the user should belong to either constants or variables
        lists.

        Note It is important to note that the user must call this function before beginning his training,
        as the debugger needs to know which types of checks the user wants to use.

        Args:
            config_path (str): The path to the configuration dict
        """
        if not (config_path is None):
            self.wandb_logger = settings.set_wandb_logger(config_path)

        if config_path is None:
            config_path = load_default_config()

        with open(config_path) as f:
            config = yaml.safe_load(f)
            # init observed_params
            params = config["debugger"]["kwargs"]["observed_params"]
            self.observed_params_update_nums.update(
                {key: 0 for key in params["variable"]}
            )
            # the constant values should be initialized with -2 which indicates that they are constants but still
            # doesn't have a value yet.
            self.observed_params_update_nums.update(
                {key: -2 for key in params["constant"]}
            )

            if config["debugger"]["kwargs"]["check_type"]:
                config = config["debugger"]["kwargs"]["check_type"]
                for debugger_config in config:
                    debugger_fn, _ = debugger_lib.get_debugger(
                        debugger_config, debugger_config["name"]
                    )
                    debugger = debugger_fn()
                    if "period" in debugger_config.keys():
                        debugger.period = debugger_config["period"]
                    if "skip_run_threshold" in debugger_config.keys():
                        debugger.config.skip_run_threshold = debugger_config[
                            "skip_run_threshold"
                        ]
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
        Turns off the debugger. This function is useful when there are testing episodes that occur during training.
        It's highly commanded to turn off the debugger if you have testing steps that occures during the training
        phase
        """
        self.training = False

    def turn_on(self):
        """
        Turns on the debugger. This function is useful when there are testing episodes that occur during training.
        This function should only be used after finishing the testing episodes that occur during th training. To call
        this function you should have already called the function turn_off
        """
        self.training = True

    def set_custom_wandb_logger(
            self,
            project,
            name,
            dir=None,
            mode=None,
            id=None,
            resume=None,
            start_method=None,
            **kwargs,
    ):
        self.wandb_logger.custom_wandb_logger(
            project, name, dir, mode, id, resume, start_method, **kwargs
        )

    def plot_wandb(self, debugger):
        if debugger.wandb_metrics and (not (self.wandb_logger is None)):
            for key, values in debugger.wandb_metrics.items():
                if values.ndim == 0:
                    self.wandb_logger.plot({key: values})
                else:
                    for value in values:
                        self.wandb_logger.plot({key: value})
            debugger.wandb_metrics = {}

    def cuda(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            for debugger in self.debuggers.values():
                debugger.device = "cuda"

    def get_device(self):
        return self.device