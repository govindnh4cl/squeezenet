import os
import argparse
from squeezenet import networks


class ArgParser(object):
    def __init__(self):
        self.parser = self._create_parser()

    def parse_args(self, args=None):
        args = self.parser.parse_args(args)
        return args

    @staticmethod
    def _create_parser():
        program_name = 'Squeezenet Training Program'
        desc = 'Program for training squeezenet with periodic evaluation.'
        parser = argparse.ArgumentParser(program_name, description=desc)

        parser.add_argument(
            '--model_dir',
            type=str,
            required=True,
            help='''Output directory for checkpoints.'''
        )
        parser.add_argument(
            '--log_dir',
            type=str,
            default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs')),
            help='''Directory for logs'''
        )

        parser.add_argument(
            '--dataset',
            type=str,
            required=True,
            default=['cifar10'],
            choices=['imagenet', 'cifar10'],
            help='Dataset to use'
        )
        parser.add_argument(
            '--network',
            type=str,
            required=True,
            choices=networks.catalogue
        )
        parser.add_argument(
            '--num_classes',
            default=10,
            type=int,
            required=True,
            help='''Number of classes (unique labels) in the dataset.
                    Ignored if using CIFAR network version.'''
        )
        parser.add_argument(
            '--num_gpus',
            default=1,
            type=int,
            required=True,
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            required=True
        )
        parser.add_argument(
            '--learning_rate', '-l',
            type=float,
            default=0.001,
            help='''Initial learning rate for ADAM optimizer.'''
        )
        parser.add_argument(
            '--batch_norm_decay',
            type=float,
            default=0.9
        )
        parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.0,
            help='''L2 regularization factor for convolution layer weights.
                    0.0 indicates no regularization.'''
        )
        # parser.add_argument(
        #     '--num_input_threads',
        #     default=1,
        #     type=int,
        #     required=True,
        #     help='''The number input elements to process in parallel.'''
        # )
        # parser.add_argument(
        #     '--shuffle_buffer',
        #     type=int,
        #     required=True,
        #     help='''The minimum number of elements in the pool of training data
        #             from which to randomly sample.'''
        # )
        parser.add_argument(
            '--seed',
            default=1337,
            type=int
        )
        parser.add_argument(
            '--max_train_epochs',
            default=1,
            type=int
        )
        # parser.add_argument(
        #     '--summary_interval',
        #     default=100,
        #     type=int
        # )
        # parser.add_argument(
        #     '--checkpoint_interval',
        #     default=100,
        #     type=int
        # )
        # parser.add_argument(
        #     '--validation_interval',
        #     default=100,
        #     type=int
        # )
        # parser.add_argument(
        #     '--keep_last_n_checkpoints',
        #     default=3,
        #     type=int
        # )
        return parser
