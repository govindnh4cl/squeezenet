import os
import time
from easydict import EasyDict
import tensorflow as tf


def _set_directories(args, cfg):
    """
    Adds directory paths in cfg and creates them if directory doesn't exist
    :param args: An Argparse object
    :param cfg: An EasyDict dictionary for configuration file
    :return: None
    """
    cfg.dir_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Directory for storing trained models
    if args.model_dir:
        cfg.dir_model = args.model_dir
    else:
        cfg.dir_model = os.path.join(cfg.dir_repo, 'models')
    os.makedirs(cfg.dir_model, exist_ok=True)
    print('Model save directory: {:s}'.format(cfg.dir_model))

    # Directory for storing log files
    if args.log_dir:
        cfg.dir_log = args.dir_log
    else:
        cfg.dir_log = os.path.join(cfg.dir_repo, 'logs')
    os.makedirs(cfg.dir_log, exist_ok=True)
    dir_tb_logs = os.path.join(cfg.dir_log, 'tensorboard')
    os.makedirs(dir_tb_logs, exist_ok=True)
    cfg.dir_tb = os.path.join(dir_tb_logs, time.strftime("%Y-%m-%d_%H-%M-%S"))  # Tensorboard directory
    print('Log dump directory: {:s}'.format(cfg.dir_log))
    print('Tensorboard directory: {:s}'.format(cfg.dir_tb))

    # Directory for storing checkpoints
    if args.ckpt_dir:
        cfg.dir_ckpt = args.ckpt_dir
    else:
        cfg.dir_ckpt = os.path.join(cfg.dir_repo, 'checkpoints')
    os.makedirs(cfg.dir_ckpt, exist_ok=True)
    print('Checkpoint directory: {:s}'.format(cfg.dir_ckpt))

    return


def _set_device(args, cfg):
    """
    Adds device specific parameters in cfg
    :param args: An Argparse object
    :param cfg: An EasyDict dictionary for configuration file
    :return: None
    """
    if args.device == 'gpu':
        cfg.device = '/gpu:0'  # Currently only support single GPU mode

        cfg.allow_memory_growth = (args.allow_memory_growth == 1)
        if cfg.allow_memory_growth:
            list_gpus = tf.config.experimental.list_physical_devices('GPU')
            if list_gpus:
                for gpu in list_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                raise Exception("tf.config.experimental.list_physical_devices('GPU') returned empty list.")
    elif args.device == 'cpu':
        cfg.device = '/cpu:0'
    else:
        raise ValueError('Unsupported args.device: {:}'.format(args.device))

    return


def get_config(args):
    """
    Returns a dictionary of configuration parameters
    :param args: An Argparse object
    :return: An EasyDict dictionary for configuration file
    """
    cfg = EasyDict(vars(args))

    # Set paths to all necessary directories
    _set_directories(args, cfg)

    # Set device to be used
    _set_device(args, cfg)

    return cfg

