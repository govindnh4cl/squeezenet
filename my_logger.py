import os
import time
import logging

import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow unnecessary prints
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning in TF's prints
    import tensorflow as tf


def _cleanup_handlers():
    """
    Delete any handlers already registered so that we can replace them with our custom logging
    This also deletes TensorFlow's default logging handler
    :return: None
    """
    logger = logging.getLogger()
    tf_logger = tf.get_logger()
    logger.handlers = []
    tf_logger.handlers = []  # This may not be


def setup_logger(log_option=0):
    """
    Adds necessary handlers in root logger
    :param log_option:
        0: No logging
        1: On screen logs
        2: 1 + File logging to logs/latest.log
        3: 2 + File logging to logs/<timestamp>.log
    :return: None
    """
    _cleanup_handlers()  # Remove any existing handlers

    logger = logging.getLogger('my_app')
    ch_formatter = logging.Formatter('{asctime:8s} {levelname:.1s} {message}', "%H:%M:%S", style='{')
    fh_formatter = logging.Formatter('{asctime:15s} {levelname:.1s} {filename}:{lineno} {message}', style='{')
    logger.setLevel(logging.DEBUG)

    if log_option >= 1:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    if log_option >= 2:
        # Latest log file for easy access
        log_file = os.path.join(log_dir, "latest.log")
        fh = logging.FileHandler(log_file, 'w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        logger.info('Dumping logs to file: {:s}'.format(log_file))

    timestamp = time.strftime('%Y.%m.%d_%H.%M.%S')
    if log_option >= 3:
        # Storing logs for archival purposes
        log_file = os.path.join(log_dir, timestamp + ".log")
        fh = logging.FileHandler(log_file, 'w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
        logger.info('Dumping logs to file: {:s}'.format(log_file))

    return None


def get_logger(name='my_app'):
    """
    Returns logger instance
    :param name: (string) name of the logger
    :return:
    """
    return logging.getLogger(name)


