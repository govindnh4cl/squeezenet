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
    # TODO: remove this extra 'train_params' directory
    cfg.directories.dir_ckpt_train = os.path.join(cfg.directories.dir_ckpt, 'train_params')
    os.makedirs(cfg.directories.dir_ckpt_train, exist_ok=True)
    logger.debug('Checkpoint train parameters directory: {:s}'.format(cfg.directories.dir_ckpt))

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

    # force all tf.function to run eagerly.
    # This would run the graph slow but would allow us to put breakpoints in case we are debugging
    assert type(cfg.hardware.force_eager) == bool  # Sanity check
    if cfg.hardware.force_eager is True:
        tf.config.experimental_run_functions_eagerly(True)

    return


def _test_dataset_params(cfg):
    """
    Test for errors in dataset parameters
    :param cfg: An EasyDict dictionary for configuration parameters
    :return: None
    """
    # ----------- Tests for CIFAR10 dataset ------------
    if cfg.dataset.dataset == 'cifar10':
        assert cfg.cifar10.num_classes == 10  # Sanity check

    # ----------- Tests for ImageNet dataset -----------
    elif cfg.dataset.dataset == 'imagenet':
        assert cfg.imagenet.num_classes == 1000  # Sanity check

        # Check if all needed file are found
        needed_files = list()

        files = dict()  # List of needed files/directories
        files['train'] = [cfg.imagenet.train_img_paths, cfg.imagenet.wnid_to_ilsvrc2012_id_path]
        files['val'] = [cfg.imagenet.val_img_base_path, cfg.imagenet.val_labels_csv]
        files['test'] = []  # TODO: implement

        if cfg.misc.mode == 'train':
            needed_files += files['train']

            if cfg.validation.enable is True:
                needed_files += files['val']

        elif cfg.misc.mode == 'eval':
            needed_files += files[cfg.eval.portion]

        for file_path in needed_files:
            if not os.path.exists(file_path):
                raise ValueError('Expected file: {:s} not found.'.format(file_path))

    # ------------ Unsupported dataset -------------------
    else:
        raise ValueError('Unsupported dataset.dataset in configuration file: {:}'.format(cfg.dataset.dataset))


def _set_dataset_params(cfg):
    """
    SAdds dataset specific parameters in cfg
    :param cfg: An EasyDict dictionary for configuration parameters
    :return: None
    """
    # Set num_classes based on dataset used
    cfg.dataset.num_classes = cfg[cfg.dataset.dataset].num_classes

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

    # Test if dataset specific parameters are correct
    _test_dataset_params(cfg)  # Test for errors in dataset parameters

    return


def _set_misc(cfg):
    if cfg.train.enable_chekpoints is True:
        if not (cfg.train.checkpoint_id in ('latest', 'none') or isinstance(cfg.train.checkpoint_id, int)):
            err_msg = "Bad configuration. model_saver.checkpoint_id should be either of  'latest', 'none' "\
                      "or an integer. Found: {:}".format(cfg.train.checkpoint_id)

            raise Exception(err_msg)
            pass
    else:
        cfg.train.checkpoint_id = 'none'  # Force set to not load from checkpoint

    return


def _set_eval(cfg):
    if cfg.misc.mode != 'eval':
        return  # Nothing to be done here

    if cfg.eval.load_from not in ('checkpoint', 'saved'):
        err_msg = "Bad configuration. eval.load_from should either be 'checkpoint' or 'saved'. Found: {:} "\
            .format(cfg.eval.load_from)

        raise Exception(err_msg)

    if not (cfg.eval.checkpoint_id in ('latest', 'none') or isinstance(cfg.eval.checkpoint_id, int)):
        err_msg = "Bad configuration. model_saver.checkpoint_id should be either of  'latest', 'none' "\
                  "or an integer. Found: {:}".format(cfg.eval.checkpoint_id)

        raise Exception(err_msg)
    else:
        cfg.eval.checkpoint_id = 'none'  # Force set to not load from checkpoint

    return


def _set_model_saver(cfg):
    if not (cfg.model_saver.checkpoint_id in ('latest', 'none') or isinstance(cfg.model_saver.checkpoint_id, int)):
        err_msg = "Bad configuration. model_saver.checkpoint_id should be either of  'latest', 'none' "\
                  "or an integer. Found: {:}".format(cfg.model_saver.checkpoint_id)

        raise Exception(err_msg)
        pass


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
    _set_eval(cfg)
    _set_model_saver(cfg)
    return cfg

