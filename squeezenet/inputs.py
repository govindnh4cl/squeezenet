from abc import abstractmethod, ABC
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from tensorflow import data

from my_logger import get_logger


def get_input_pipeline(cfg, purpose, portion):
    """
    A factory for instantiating the pipeline object
    :param cfg:
    :param purpose: 'train', 'inference'.
        'train': For training the network
        'inference': For validation, testing, deployment, inference etc.
    :param portion: 'train', 'validation', 'test', None
        Which portion of the dataset to load.
            'train': Take samples from train split
            'validation': Take samples from validation split
            'test': Take samples from test split. This does not have labels.
    :return: An derived class object of class Pipeline
    """
    if cfg.dataset.dataset == 'imagenet':
        from squeezenet.input_imagenet import InputImagenet  # Not done on top to avoid circular dependency
        pipeline = InputImagenet(cfg, purpose, portion)
    elif cfg.dataset.dataset == 'cifar10':
        raise NotImplementedError

    return pipeline


class Pipeline(ABC):
    """
    Base class of all data input pipelines
    """
    def __init__(self, cfg):
        self._cfg = cfg
        self._logger = get_logger()
        self._dataset = None
        self._count = 0  # count of total samples in this pipeline

    def get_dataset(self):
        """

        :return: The reference to tf.data.Dataset class object
        """
        return self._dataset

    def __len__(self):
        """
        Returns the count of total samples in this pipeline
        :return: Integer count
        """
        return int(self._count)

