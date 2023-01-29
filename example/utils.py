import copy
from hive import envs
from hive.utils.registry import get_parsed_args
from hive.utils import experiment, loggers, schedule, utils
from hive import agents as agent_lib
from hive.runners.single_agent_loop import SingleAgentRunner
from debugger.debugger_factory import DebuggerFactory


class DebuggerRunner(SingleAgentRunner):
    def __init__(self, environment, agent, logger, debugger, experiment_manager, train_steps, test_frequency,
                 test_episodes, stack_size, max_steps_per_episode=27000):
        super().__init__(environment, agent, logger, experiment_manager, train_steps, test_frequency, test_episodes,
                         stack_size, max_steps_per_episode)
        self._debugger = debugger


def set_up_experiment(config):
    """Returns a :py:class:`SingleAgentRunner` object based on the config and any
    command line arguments.

    Args:
        config: Configuration for experiment.
    """

    args = get_parsed_args(
        {
            "seed": int,
            "train_steps": int,
            "test_frequency": int,
            "test_episodes": int,
            "max_steps_per_episode": int,
            "stack_size": int,
            "resume": bool,
            "run_name": str,
            "save_dir": str,
        }
    )
    config.update(args)
    full_config = utils.Chomp(copy.deepcopy(config))

    if "seed" in config:
        utils.seeder.set_global_seed(config["seed"])

    environment_fn, full_config["environment"] = envs.get_env(
        config["environment"], "environment"
    )
    environment = environment_fn()
    env_spec = environment.env_spec

    # Set up loggers
    logger_config = config.get("loggers", {"name": "NullLogger"})
    if logger_config is None or len(logger_config) == 0:
        logger_config = {"name": "NullLogger"}
    if isinstance(logger_config, list):
        logger_config = {
            "name": "CompositeLogger",
            "kwargs": {"logger_list": logger_config},
        }

    logger_fn, full_config["loggers"] = loggers.get_logger(logger_config, "loggers")
    logger = logger_fn()

    # Set up debugger
    # TODO: revise this !!
    debugger = None
    if "debugger" in config:
        debugger_config = config["debugger"]
        # TODO: merge these two lines;
        debugger = DebuggerFactory(debugger_config["kwargs"]["check_type"])

    agent_fn, full_config["agent"] = agent_lib.get_agent(config["agent"], "agent")
    # TODO inject debugger in all the agents configs
    agent = agent_fn(
        observation_space=env_spec.observation_space[0],
        action_space=env_spec.action_space[0],
        stack_size=config.get("stack_size", 1),
        logger=logger,
        debugger=debugger
    )

    # Set up experiment manager
    saving_schedule_fn, full_config["saving_schedule"] = schedule.get_schedule(
        config["saving_schedule"], "saving_schedule"
    )
    experiment_manager = experiment.Experiment(
        config["run_name"], config["save_dir"], saving_schedule_fn()
    )
    # experiment_manager is used to save various components (like logger, agent, ...)  in one experiment.
    # TODO: maybe we need to add the debugger here
    experiment_manager.register_experiment(
        config=full_config,
        logger=logger,
        agents=agent,
    )
    # Set up runner
    runner = DebuggerRunner(
        environment,
        agent,
        logger,
        debugger,
        experiment_manager,
        config.get("train_steps", -1),
        config.get("test_frequency", -1),
        config.get("test_episodes", 1),
        config.get("stack_size", 1),
        config.get("max_steps_per_episode", 1e9),
    )
    if config.get("resume", False):
        runner.resume()

    return runner
