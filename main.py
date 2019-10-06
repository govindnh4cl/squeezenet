import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)  # Disable FutureWarning in TF's prints
    import tensorflow as tf

from my_logger import setup_logger, get_logger
from squeezenet.arg_parsing import parse_args
from squeezenet.develop_squeezenet import DevelopSqueezenet


def main():
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
