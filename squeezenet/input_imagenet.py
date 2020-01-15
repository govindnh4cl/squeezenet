import os
import numpy as np
import tensorflow as tf
from tensorflow import data

from squeezenet.input_imagenet_base import InputImagenetBase


class InputImagenetTrain(InputImagenetBase):
    """ Base class for Imagenet dataset input pipeline """

    def __init__(self, cfg):
        InputImagenetBase.__init__(self, cfg)  # Base class

        self._parse_img_paths_labels()  # Populate the variables self._img_paths and self._img_labels

        return

    def _parse_img_paths_labels(self):
        """
        Virtual function implementation.
        :return: None. Updates self._img_paths and self._img_labels member variables
        """
        with open(self._cfg.imagenet.train_img_paths, 'r') as fp:
            self._img_paths = [line for line in fp]

        # Sanity check
        if len(self._img_paths) != 1_281_167:
            raise Exception('Found incorrect training img paths required for Imagenet dataset. '
                            'Found paths: {:d}, expected paths: 1,281,167. '
                            'Please verify the contents of {:s} file.'.
                            format(len(self._img_paths), self._cfg.imagenet.train_img_paths))

        # Get corresponding integer class ID (1 to 1000) for each image in self._img_paths
        self._img_labels = [self._wnid_to_ilsvrc2012_id[os.path.basename(os.path.dirname(img_path))]
                            for img_path in self._img_paths]

        return




