import sys
import os
import argparse
import tensorflow as tf

repo_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(repo_path)

from my_logger import setup_logger, get_logger
setup_logger(log_option=1)

from squeezenet.arg_parsing import parse_args
from squeezenet.develop_squeezenet import DevelopSqueezenet


def main():
    logger = get_logger()
    args = parse_args()
    sqz = DevelopSqueezenet(args)
    sqz.load_checkpointables('latest')  # Load checkpoint

    concrete_fn = sqz.net.call.get_concrete_function(
        batch_x=tf.TensorSpec([None, 224, 224, 3], tf.float32),
        training=False)

    logger.info('Saving the model in directory: {:s}'.format(sqz.cfg.directories.dir_model))
    tf.saved_model.save(sqz.net, sqz.cfg.directories.dir_model)  # Save model
    return


if __name__ == '__main__':
    main()
