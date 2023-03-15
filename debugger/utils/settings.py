import os
import logging
from configparser import ConfigParser
from pathlib import Path
from debugger.utils.wandb_logger import WandbLogger
import yaml

# Conventional constants
LOG_DIR = "logs"


# Standard messages
def load_messages(messages_section="Messages"):
    """
    Loads the messages from the `messages.properties` file.

    Args:
        messages_section (str): The section in the `messages.properties` file to load messages from. Default is
        `Messages`.

    Returns:
        dict: A dictionary containing the messages from the specified section of the `messages.properties` file.
    """
    messages_fpath = os.path.join(
        os.path.join(Path(__file__).parent, "config"), "messages.properties"
    )
    config = ConfigParser()
    config.read(messages_fpath)
    return dict(config[messages_section])


# Logging
def build_log_file_path(app_path, app_name):
    """
    Builds the file path for the log file.

    Args:
        app_path (str): The path of the application.
        app_name (str): The name of the application.

    Returns:
            str: The file path for the log file.
    """
    log_dir_path = os.path.join(app_path, LOG_DIR)
    if not (os.path.exists(log_dir_path)):
        os.makedirs(log_dir_path)
    return os.path.join(log_dir_path, f"{app_name}.log")


def file_logger(file_path, app_name):
    """
    Creates a logger object that writes logs to both a file and the console.

    Args:
        file_path (str): The file path for the log file.
        app_name (str): The name of the application.

    Returns:
        logging.Logger: A logger object that writes logs to both a file and the console.
    """
    logger = logging.getLogger(f"TheDeepChecker: {app_name} Logs")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def set_logger():
    app_path = str(Path.cwd())
    log_fpath = build_log_file_path(app_path, "logger")
    return file_logger(log_fpath, "logger")


def set_wandb_logger(config_path=None):
    if config_path is None:
        config_path = load_default_config()

    with open(config_path) as f:
        config = yaml.safe_load(f)
        return WandbLogger(
            project=config["debugger"]["wand_logger"]["project"],
            name=config["debugger"]["wand_logger"]["name"],
        )


def react(logger, messages, fail_on=False):
    """
    Reacts to the provided `messages` by either raising an exception or logging a warning, depending on the value of
     `fail_on`.

    Args:
        logger (logger): the logger that will print the data
        messages (list): list of error messages to be displayed
        fail_on (bool): if True it raises an exception otherwise it only displays the error
    """
    if len(messages) > 0:
        for message in messages:
            if fail_on:
                logger.error(message)
                raise Exception(message)
            else:
                logger.warning(message)


def load_default_config():
    return os.path.join(
        os.path.join(Path(__file__).parent, "config"), "default_debugger.yml"
    )
