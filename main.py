import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning in TF's prints
    import tensorflow as tf


from squeezenet.utils import set_random_seed
set_random_seed(0)  # TODO: Take it from the config file?

from my_logger import setup_logger, get_logger
from squeezenet.arg_parsing import parse_args
from squeezenet.develop_squeezenet import DevelopSqueezenet


def main():
    # log_option:
    #    0: No logging
    #    1: On screen logs
    #    2: 1 + File logging to logs/latest.log
    #    3: 2 + File logging to logs/<timestamp>.log
    setup_logger(log_option=2)
    logger = get_logger()

    args = parse_args()
    dev = DevelopSqueezenet(args)

    try:
        dev.run()
    except Exception as e:
        logger.exception(str(e))

    return


if __name__ == '__main__':
    main()
