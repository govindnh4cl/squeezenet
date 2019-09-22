import os
import time
import toml
from easydict import EasyDict
import tensorflow as tf

from my_logger import get_logger


def _parse_cfg(file_path):
    """
    Parses configuration file to a dictionary
    :param file_path: Path to .toml configuration file
    :return: An EasyDict object with configuration parameters
    """
    cfg = EasyDict(toml.load(file_path))
    return cfg


def _set_directories(cfg):
    """
    Sets up directory paths in cfg and creates them if directory doesn't exist
    :param cfg: An EasyDict dictionary for configuration parameters
    :return: None
    """
    logger = get_logger()

    # Repository path
    cfg.directories.dir_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Convert all directory paths to absolute paths
    for directory in cfg.directories.keys():
        if not os.path.isabs(directory):
            cfg.directories[directory] = os.path.join(cfg.directories.dir_repo, cfg.directories[directory])

    # Create directories if they don't exist
    os.makedirs(cfg.directories.dir_model, exist_ok=True)
    logger.debug('Model save directory: {:s}'.format(cfg.directories.dir_model))
    os.makedirs(cfg.directories.dir_log, exist_ok=True)
    logger.debug('Log dump directory: {:s}'.format(cfg.directories.dir_log))
    os.makedirs(cfg.directories.dir_tb_home, exist_ok=True)
    # Tensorboard directory
    cfg.directories.dir_tb = os.path.join(cfg.directories.dir_tb_home, time.strftime("%Y-%m-%d_%H-%M-%S"))
    logger.debug('Tensorboard directory: {:s}'.format(cfg.directories.dir_tb))
    os.makedirs(cfg.directories.dir_ckpt, exist_ok=True)
    logger.debug('Checkpoint directory: {:s}'.format(cfg.directories.dir_ckpt))

    return


def _set_hardware(cfg):
    """
    Sets up device specific parameters in cfg
    :param cfg: An EasyDict dictionary for configuration parameters
    :return: None
    """
    if cfg.hardware.device == 'gpu':
        assert cfg.hardware.num_gpu == 1  # Currently only support single GPU mode
        cfg.hardware.device = '/gpu:0'  # Convert to a string representation that TF understands

        assert type(cfg.hardware.allow_memory_growth) is bool  # Sanity check
        if cfg.hardware.allow_memory_growth:
            # Set GPU to only allocate memory as needed
            list_gpus = tf.config.experimental.list_physical_devices('GPU')
            if list_gpus:
                for gpu in list_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                raise Exception("tf.config.experimental.list_physical_devices('GPU') returned empty list.")
    elif cfg.hardware.device == 'cpu':
        cfg.hardware.device = '/cpu:0'  # Convert to a string representation that TF understands
    else:
        raise ValueError('Unsupported hardware.device: {:} in configuration file'.format(cfg.hardware.device))

    return


def _set_dataset_params(cfg):
    """
    SAdds dataset specific parameters in cfg
    :param cfg: An EasyDict dictionary for configuration parameters
    :return: None
    """
    if cfg.dataset.dataset not in ('cifar10', 'imagenet'):
        raise ValueError('Unsupported dataset.dataset in configuration file: {:}'.format(cfg.dataset.dataset))

    # Set num_classes based on dataset used
    if cfg.dataset.dataset == 'cifar10':
        cfg.dataset.num_classes = 10
    elif cfg.dataset.dataset == 'imagenet':
        cfg.dataset.num_classes = 1000

    if cfg.misc.phase not in ('train', 'test'):
        raise ValueError('Unsupported misc.phase: {:}'.format(cfg.misc.phase))
    elif cfg.misc.phase == 'train':
        cfg.train.enable = True                         # Enable train
        assert type(cfg.validation.enable) is bool      # Take from config file
        cfg.test.enable = False                         # Disable test
    elif cfg.misc.phase == 'test':
        cfg.train.enable = False                        # Disable train
        cfg.validation.enable = False                   # Disable validation
        cfg.test.enable = True                          # Enable test

    # If value is -1 in config, then use the batch_size from training configuration
    if cfg.validation.enable and cfg.validation.batch_size == -1:
        cfg.validation.batch_size = cfg.train.batch_size
    if cfg.test.enable and cfg.test.batch_size == -1:
        cfg.test.batch_size = cfg.train.batch_size

    # Error check on batch_size
    if cfg.train.enable and cfg.train.batch_size % 2 != 0:
        raise ValueError('Train batch size {:} is not multiple of 2.'.format(cfg.train.batch_size))
    if cfg.validation.enable and cfg.validation.batch_size % 2 != 0:
        raise ValueError('Validation batch size {:} is not multiple of 2.'.format(cfg.validation.batch_size))
    if cfg.test.enable and cfg.test.batch_size % 2 != 0:
        raise ValueError('Test batch size {:} is not multiple of 2.'.format(cfg.test.batch_size))

    return


def get_config(args):
    """
    Returns a dictionary of configuration parameters
    :param args: An Argparse object
    :return: An EasyDict dictionary for configuration parameters
    """
    if not os.path.exists(args.cfg_file):
        raise FileNotFoundError('The configuration file: {:s} does not exist.'.format(args.cfg_file))

    cfg = _parse_cfg(args.cfg_file)

    _set_directories(cfg)  # Set paths to all necessary directories
    _set_hardware(cfg)  # Set device to be used
    _set_dataset_params(cfg)  # Set dataset and train/val/test phase related parameters

    return cfg

