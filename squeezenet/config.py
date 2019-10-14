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
    if cfg.train.enable_train_chekpoints is True:
        cfg.directories.dir_ckpt_train = os.path.join(cfg.directories.dir_ckpt, 'train_params')
        os.makedirs(cfg.directories.dir_ckpt_train, exist_ok=True)
        logger.debug('Checkpoint train parameters directory: {:s}'.format(cfg.directories.dir_ckpt))
    if cfg.train.enable_save_best_model is True:
        cfg.directories.dir_ckpt_save_model = os.path.join(cfg.directories.dir_ckpt, 'save_best_model')
        os.makedirs(cfg.directories.dir_ckpt_save_model, exist_ok=True)
        logger.debug('Checkpoint save best model directory: {:s}'.format(cfg.directories.dir_ckpt))

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

    # Hold parameters for portions of datasets
    cfg.dataset.train = EasyDict()
    cfg.dataset.val = EasyDict()
    cfg.dataset.test = EasyDict()

    # Enable/Disable loading of certain portion of dataset based on config file
    if cfg.misc.mode not in ('train', 'eval'):
        raise ValueError('Unsupported misc.mode: {:}'.format(cfg.misc.mode))
    elif cfg.misc.mode == 'train':
        cfg.dataset.train.enable = True                 # Enable loading of train set
        assert type(cfg.validation.enable) is bool      # Sanity check
        cfg.dataset.val.enable = cfg.validation.enable  # Enable/Disable loading of val set based on config
        cfg.dataset.test.enable = False                 # Disable loading of test set
    elif cfg.misc.mode == 'eval':
        assert cfg.eval.portion in ('train', 'val', 'test')     # Sanity check
        cfg.dataset.train.enable = (cfg.eval.portion == 'train')
        cfg.dataset.val.enable = (cfg.eval.portion == 'val')
        cfg.dataset.test.enable = (cfg.eval.portion == 'test')

    # Set batch_size parameters for dataset portions
    # FYI: we store batch_size in two places for convenience:
    #   cfg.dataset.<phase>.batch_size
    #   cfg.<phase>.batch_size
    # where <phase> is train, validation or test
    if cfg.misc.mode == 'train':
        if cfg.dataset.train.enable:
            cfg.dataset.train.batch_size = cfg.train.batch_size

        if cfg.dataset.val.enable:
            if cfg.validation.batch_size == -1:
                # use the same batch_size as for training
                cfg.validation.batch_size = cfg.train.batch_size
                cfg.dataset.val.batch_size = cfg.train.batch_size
            else:
                cfg.dataset.val.batch_size = cfg.validation.batch_size
    elif cfg.misc.mode == 'eval':
        if cfg.dataset.train.enable:
            cfg.dataset.train.batch_size = cfg.eval.batch_size

        if cfg.dataset.val.enable:
            cfg.dataset.val.batch_size = cfg.eval.batch_size

        if cfg.dataset.test.enable:
            cfg.dataset.test.batch_size = cfg.eval.batch_size

    return


def _set_misc(cfg):
    if cfg.validation.enable is False:
        if cfg.train.enable_save_best_model == 'val_loss':
            raise ValueError('Bad config parameters: validation.enable is false and '
                             'train.enable_save_best_model is still set to "val_loss". '
                             'Either set validation.enable to true or change train.enable_save_best_model '
                             'to something else.')


def get_config(args):
    """
    Returns a dictionary of configuration parameters
    :param args: An Argparse object
    :return: An EasyDict dictionary for configuration parameters
    """
    if not os.path.exists(args.cfg_file):
        raise FileNotFoundError('The configuration file: {:s} does not exist.'.format(args.cfg_file))

    cfg = _parse_cfg(args.cfg_file)

    _set_misc(cfg)
    _set_directories(cfg)  # Set paths to all necessary directories
    _set_hardware(cfg)  # Set device to be used
    _set_dataset_params(cfg)  # Set dataset and train/val/test set related parameters

    return cfg

