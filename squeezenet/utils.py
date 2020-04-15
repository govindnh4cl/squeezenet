import numpy as np
import tensorflow as tf

from my_logger import get_logger

logger = get_logger()


def set_random_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


def load_saved_model(dir_save_model):
    logger.info("Loading model from directory: {:s}".format(dir_save_model))
    if not tf.saved_model.contains_saved_model(dir_save_model):
        raise OSError("Model directory: {:s} does not contain a saved model.")
    else:
        net = tf.saved_model.load(dir_save_model)

    return net

