import os
from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow import data
import json

from squeezenet.inputs import Pipeline


class InputImagenetBase(Pipeline):
    """ Base class for Imagenet dataset input pipeline """
    def __init__(self, cfg):
        Pipeline.__init__(self, cfg)

        self._wnid_to_ilsvrc2012_id = self._get_wnid_to_ilsvrc2012_id()

        # These variables would be populated by derived classes
        self._img_paths = None  # List of image paths
        self._img_labels = None  # List of numerical class labels [1, 1000]

        self._INPUT_SIZE = 224  # Input image resolution for Imagenet dataset

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

    def _preprocess(self):
        """
        Resize, typecast etc.
        :return:
        """
        pass

    def _normalize_image(self):
        """

        :return:
        """
        # TODO: implementation
        return


class InputImagenetTrain(InputImagenetBase):
    """ Base class for Imagenet dataset input pipeline """

    def __init__(self, cfg, portion):
        """

        :param cfg:
        :param portion: 'train', 'validation' or 'test'
        """
        InputImagenetBase.__init__(self, cfg)  # Base class
        self._portion = portion
        self._batch_size = self._cfg[self._portion].batch_size
        self._parse_img_paths_labels()  # Populate the variables self._img_paths and self._img_labels
        assert len(self._img_paths) == len(self._img_labels)
        self._count = len(self._img_paths)  # count of samples

        # Whether this pipeline is for learning new weights
        self._do_backprop = (self._portion == 'train') and (self._cfg.misc.mode == 'train')
        # Whether to shuffle the samples. Shuffling happens each epoch
        self._do_shuffle = self._do_backprop
        # Whether to perturb the image sample as data augmentation
        self._do_perturb = self._do_backprop

        with tf.device('/cpu:0'):
            self._dataset = self._prepare_dataset()

        return

    def _parse_img_paths_labels(self):
        """
        Virtual function implementation.
        :return: None. Updates self._img_paths and self._img_labels member variables
        """
        with open(self._cfg.imagenet.train_img_paths, 'r') as fp:
            self._img_paths = fp.read().splitlines()

        # Sanity check
        if len(self._img_paths) != 1_281_167:
            raise Exception('Found incorrect training img paths required for Imagenet dataset. '
                            'Found paths: {:d}, expected paths: 1,281,167. '
                            'Please verify the contents of {:s} file.'.
                            format(len(self._img_paths), self._cfg.imagenet.train_img_paths))

        if self._portion != 'test':
            # Get corresponding integer class ID (1 to 1000) for each image in self._img_paths
            self._img_labels = [self._wnid_to_ilsvrc2012_id[os.path.basename(os.path.dirname(img_path))]
                                for img_path in self._img_paths]
        else:
            # Populate dummy labels for the test set since we don't get them for Imagenet
            self._img_labels = [0] * len(self._img_paths)  # 0 as the dummy label

        return

    def _generator(self):
        indices = np.arange(self._count)

        # Shuffle the samples. A new ordering each epoch
        if self._do_shuffle is True:
            np.random.shuffle(indices)

        for i in indices:
            x = tf.io.decode_jpeg(tf.io.read_file(self._img_paths[i]), channels=3)
            y = self._img_labels[i]

            # import pdb
            # pdb.set_trace()

            yield x, y

        return

    def _prepare_dataset(self):

        # Images returned by the generator can have uneven resolution
        dataset = tf.data.Dataset.from_generator(self._generator,
                                                 output_types=(tf.uint8, tf.int64),
                                                 output_shapes=(tf.TensorShape([None, None, 3]),
                                                                tf.TensorShape([])))

        # Preprocess samples
        dataset = dataset.map(lambda x, y: (self._preprocess_x(x), self._preprocess_y(y)))

        return dataset

    def _preprocess_x(self, x):
        """
        Preprocess image
        :return:
        """
        # TODO: Is this really necessary
        x = tf.cast(x, tf.float32)  # Cast datatype from tf.uint8 to tf.float32

        if self._cfg.model.data_format == 'channels_last':
            # Normalize image. Means (Blue, Green, Red): 104, 117, 123
            # x[:, :, :, 0] -= 123  # Red
            # x[:, :, :, 1] -= 117  # Green
            # x[:, :, :, 2] -= 104  # Blue

            x = tf.math.subtract(x, [123, 117, 104])  # RGB channel order assumed
            return x

            # TODO: Scale the images to (-1, +1) ?

            # Resize and crop image
            small_edges = min(x.shape[1], x.shape[2])  # A 1-D array
            s_upon_h = small_edges / x.shape[:, 1]
            s_upon_w = small_edges / x.shape[:, 2]

            boxes = [(1 - s_upon_h) / 2, (1 - s_upon_w) / 2, (1 + s_upon_h) / 2, (1 + s_upon_w) / 2]

            x = tf.image.crop_and_resize(
                x,
                boxes,
                tf.range(x.shape[0]),
                (224, 224))

            # Randomly flip left-right. Done only for training phase
            if self._do_perturb:
                x = tf.image.random_flip_left_right(x)

        else:
            raise NotImplementedError
            x[:, 0, :, :] -= 123
            x[:, 1, :, :] -= 117
            x[:, 2, :, :] -= 104

        return x

    def _preprocess_y(self, y):
        """
        Preprocess labels
        :param y: A tensor of shape (None, 1). Contains integer class label (1, 1000)
        :return: A tensor of shape (None, 1000)
        """
        # Convert labels to one-hot
        y = tf.one_hot(indices=y - 1, depth=1000)

        return y
