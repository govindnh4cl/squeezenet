from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import data
import json


class InputImagenetBase:
    """ Base class for Imagenet dataset input pipeline """
    def __init__(self, cfg):
        self._cfg = cfg

        self._wnid_to_ilsvrc2012_id = self._get_wnid_to_ilsvrc2012_id()

        # These variables would be populated by derived classes
        self._img_paths = None  # List of image paths
        self._img_labels = None  # List of numerical class labels [1, 1000]



        return

    def _get_wnid_to_ilsvrc2012_id(self):
        """
        Sets dictionary mapping WNID (e.g. 'n01440764') to class label (int in range 1 to 1000)
        # This is needed to get class labels of training images from their paths
        :return: dictionary
        """
        with open(self._cfg.imagenet.wnid_to_ilsvrc2012_id_path, 'r') as fp:
            d = json.load(fp)

        return d

    @abstractmethod
    def _parse_img_paths_labels(self):
        raise NotImplementedError

    def _normalize_image(self):
        """

        :return:
        """
        # TODO: implementation
        return


