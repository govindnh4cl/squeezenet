import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning in TF's prints
    import tensorflow as tf

from my_logger import setup_logger, get_logger
from squeezenet.train_squeezenet import run

if __name__ == '__main__':
    setup_logger(log_option=2)
    logger = get_logger()

    try:
        run()
    except Exception as e:
        logger.exception(str(e))
