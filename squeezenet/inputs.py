from abc import abstractmethod, ABC
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from tensorflow import data

from my_logger import get_logger

def get_input_pipeline(cfg, portion):
    """
    A factory for instantiating the pipeline object
    :param cfg:
    :param portion: Portion of entire dataset to use. Supported: 'train', 'val', 'test'
    :return: An derived class object of class Pipeline
    """
    if cfg.dataset.dataset == 'imagenet':
        if portion == 'train':
            from squeezenet.input_imagenet import InputImagenetTrain  # Not done on top to avoid circular dependency
            pipeline = InputImagenetTrain(cfg, portion='train')
        elif portion == 'val':
            raise NotImplementedError
        elif portion == 'test':
            raise NotImplementedError
        else:
            raise Exception('Unsupported dataset portion: {:}'.format(portion))

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

    def get_count(self):
        """
        Returns the count of total samples in this pipeline
        :return: Integer count
        """
        return int(self._count)

