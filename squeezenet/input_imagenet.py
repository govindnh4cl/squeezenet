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
        indices = tf.range(self._count)

        # Enable for debugging purposes
        if 0:
            tf.random.set_seed(0)

            # Shuffle the samples. A new ordering each epoch
            if self._do_shuffle is True:
                indices = tf.random.shuffle(indices, seed=0)

            limit = 16384  # Just use a small set
            indices = indices[:limit]
        else:
            # Shuffle the samples. A new ordering each epoch
            if self._do_shuffle is True:
                indices = tf.random.shuffle(indices, seed=1337)

        for i in indices:
            x = self._img_paths[i]
            y = self._img_labels[i]

            # x = self._preprocess_x(x)
            y = self._preprocess_y(y)

            yield x, y

        return

    def _prepare_dataset(self):

        # Images returned by the generator can have uneven resolution
        dataset = tf.data.Dataset.from_generator(self._generator,
                                                 output_types=(tf.dtypes.string, tf.dtypes.float32),
                                                 output_shapes=(tf.TensorShape([]),
                                                                tf.TensorShape([1000])))
        # Prefetch samples using a separate thread for faster file-read
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Preprocess samples
        dataset = dataset.map(
            lambda x, y: (tf.py_function(func=self._preprocess_x_non_graph, inp=[x], Tout=tf.dtypes.float32), y),
            num_parallel_calls=4)

        # Batching should only be performed after the preprocessing makes all samples same shape
        dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)

        dataset = dataset.map(
            lambda x, y: (self._preprocess_x_with_graph(x), y),
            num_parallel_calls=4)

        return dataset

    # Don't use @tf.function for this as due to uneven image resolutions, this function
    # cannot be converted to a graph
    def _preprocess_x_non_graph(self, x):
        """
        Preprocess image
        Contains processing that cannot be converted to a graph as the input image resolution is different
        across dataset
        :param x: String containing path of image

        :return: tensor of shape (224, 224, 3)
        """
        # Read image from filesystem
        x = tf.io.decode_jpeg(tf.io.read_file(x), channels=3)

        if self._cfg.model.data_format == 'channels_last':
            # Crop image
            ht, wd = x.shape[0], x.shape[1]
            small_edge = min(ht, wd)
            x_min = int((wd - small_edge) / 2)
            x_max = int(x_min + small_edge)
            y_min = int((ht - small_edge) / 2)
            y_max = int(y_min + small_edge)

            cropped_x = x[y_min:y_max, x_min:x_max, :]  # Cropping operation

            # Resize image to common size.
            # This converts the dtype from uint8 to float32
            x = tf.image.resize(cropped_x, (224, 224))

        else:
            raise NotImplementedError
            x[:, 0, :, :] -= 123
            x[:, 1, :, :] -= 117
            x[:, 2, :, :] -= 104

        return x

    @tf.function
    def _preprocess_x_with_graph(self, x):
        """
        Preprocess image
        Contains processing that can be converted to a graph due to predetermined input/output shapes
        across dataset

        :return: tensor of same shape as x
        """
        # As per documentation, TF traces all map() functions to create their graph
        # During tracing, the shape of x is not yet known. Conceptually we're certain
        # that all inputs to this function are going to be 224x224 images. Hence, we
        # must set it here so that the returned value of this function can have a known
        # shape during tracing.
        x.set_shape((None, 224, 224, 3))

        # Normalize image. Means (Blue, Green, Red): 104, 117, 123
        x = tf.math.subtract(x, [123, 117, 104])  # RGB channel order assumed
        # Scale the images to range about [-1, +1]. Some elements could go outside this range
        x = tf.math.divide(x, 127.)

        # Randomly flip left-right. Done only for training phase
        if self._do_perturb:
            x = tf.image.random_flip_left_right(x)

        return x

    def _preprocess_y(self, y):
        """
        Preprocess labels
        :param y: A tensor of shape (None, 1). Contains integer class label (1, 1000)
        :return: A tensor of shape (None, 1000)
        """
        # Convert labels to one-hot
        y = tf.one_hot(indices=y - 1, depth=1000, dtype=tf.dtypes.float32)

        return y
