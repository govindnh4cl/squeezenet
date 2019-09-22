import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg-file',
        type=str,
        help='Path to .toml configuration file'
    )

    args = parser.parse_args()
    return args
