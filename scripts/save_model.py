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

    dummy_img = tf.random.uniform(shape=(1, 224, 224, 3), dtype=tf.float32)
    out = sqz.net(dummy_img)
    logger.info('Dummy output of shape: {:}'.format(out.shape))

    logger.info('Saving the model in directory: {:s}'.format(sqz.cfg.directories.dir_model))
    tf.saved_model.save(sqz.net, sqz.cfg.directories.dir_model)  # Save model
    return


if __name__ == '__main__':
    main()
