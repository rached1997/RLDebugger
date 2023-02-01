# TODO: remove unused functions
import os
import logging
import yaml
from fastnumbers import fast_real
from configparser import ConfigParser
from pathlib import Path

# Conventional constants
LOG_DIR = 'logs'


# Standard messages
def load_messages(messages_section='Messages'):
    messages_fpath = os.path.join(os.path.join(Path(__file__).parent, 'config'), 'messages.properties')
    config = ConfigParser()
    config.read(messages_fpath)
    return dict(config[messages_section])


# Logging
def build_log_file_path(app_path, app_name):
    log_dir_path = os.path.join(app_path, LOG_DIR)
    if not (os.path.exists(log_dir_path)):
        os.makedirs(log_dir_path)
    return os.path.join(log_dir_path, f'{app_name}.log')


def file_logger(file_path, app_name):
    logger = logging.getLogger(f'TheDeepChecker: {app_name} Logs')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def console_logger(app_name):
    logger = logging.getLogger(f'TheDeepChecker: {app_name} Logs')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
