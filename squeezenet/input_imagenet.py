import os
from abc import abstractmethod
import json
import numpy as np
import tensorflow as tf
import pandas as pd

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


class InputImagenet(InputImagenetBase):
    """ Base class for Imagenet dataset input pipeline """

    def __init__(self, cfg, purpose, portion):
        """

        :param cfg:
        :param purpose: 'train', 'inference'.
            'train': For training the network
            'inference': For validation, testing, deployment, inference etc.
        :param portion: 'train', 'val', 'test', None
            Which portion of the dataset to load.
                'train': Take samples from train split
                'validation': Take samples from validation split
                'test': Take samples from test split. This does not have labels.
            None: No samples to be loaded. Used during deployment.
        """
        InputImagenetBase.__init__(self, cfg)  # Base class
        self._purpose = purpose
        self._portion = portion

        # Sanity check
        if self._purpose == 'train':
            assert self._portion != 'test'  # Can't train with test set as no ground truth available for Imagenet
            assert self._portion is not None  # Must specify a source to load the training samples from

        self._batch_size = self._cfg.dataset[self._portion].batch_size
        self._img_paths, self._img_labels = self._load_input_data()  # Load image paths and corresponding labels
        self._count = len(self._img_paths)  # count of samples

        # Whether this pipeline is for learning new weights
        self._do_backprop = (self._purpose == 'train')
        # Whether to shuffle the samples. Shuffling happens each epoch
        self._do_shuffle = self._do_backprop  # Only shuffle when learning
        # Whether to perturb the image sample as data augmentation
        self._do_perturb = self._do_backprop  # Augmentation needed only while learning

        self._dataset = self._prepare_dataset()  # An object of class tf.data.Dataset

        self._arrange_samples()  # May shuffle if configured

        return

    def _arrange_samples(self):
        indices = tf.range(len(self))

        # Enable for debugging purposes. Limits sample count to a small number.
        if 0:
            tf.random.set_seed(0)
            limit = min(4096, len(indices))  # Just use a small set
            indices = indices[:limit]
            self._count = limit  # The number of samples

        # Shuffle the samples
        if self._do_shuffle is True:
            indices = tf.random.shuffle(indices, seed=1337)

        self._img_paths = [self._img_paths[x] for x in indices]
        self._img_labels = [self._img_labels[x] for x in indices]

        return indices

    def _load_training_set(self):
        """
        Loads training set image paths and corresponding labels into memory.
        :return: list of img paths (string), list of class ID (integer) 1 to 1000
        """
        with open(self._cfg.imagenet.train_img_paths, 'r') as fp:
            img_paths = fp.read().splitlines()

        # Sanity check. Imagenet should have 1,281,167 training images
        if len(img_paths) != 1_281_167:
            raise Exception('Found incorrect training img paths required for Imagenet dataset. '
                            'Found paths: {:d}, expected paths: 1,281,167. '
                            'Please verify the contents of {:s} file.'.
                            format(len(img_paths), self._cfg.imagenet.train_img_paths))

        # Get corresponding integer class ID (1 to 1000) for each image in self._img_paths
        img_labels = [self._wnid_to_ilsvrc2012_id[os.path.basename(os.path.dirname(img_path))]
                      for img_path in img_paths]

        return img_paths, img_labels

    def _load_validation_set(self):
        """
        Loads training set image paths and corresponding labels into memory.
        :return: list of img paths (string), list of class ID (integer) 1 to 1000
        """
        count_samples = 50_000  # Expected sample count in validation set

        df = pd.read_csv(self._cfg.imagenet.val_labels_csv)
        df = df.set_index('ImageId', drop=True)

        # List of image names in validation set
        img_names = [x for x in os.listdir(self._cfg.imagenet.val_img_base_path) if x.endswith('.JPEG')]
        # Sanity check. Imagenet should have 50k validation images
        if not (len(df) == len(img_names) == count_samples):
            raise Exception('Found incorrect validation set img samples required for Imagenet dataset. '
                            'Expected samples: {:d}. '
                            'Found images: {:d} Found ground truth: {:d}. '
                            'Please verify the contents of val directory: {:s} and ground truth file" {:s}.'.
                            format(count_samples, len(img_names), len(df),
                                   self._cfg.imagenet.val_img_base_path, self._cfg.imagenet.val_labels_csv))

        img_paths = [''] * count_samples
        img_labels = [0] * count_samples

        for i, img_name in enumerate(img_names):
            img_id = os.path.splitext(img_name)[0]  # Image name without extension
            img_paths[i] = os.path.join(self._cfg.imagenet.val_img_base_path, img_name)

            wnid = df.at[img_id, 'PredictionString'].split(' ')[0]  # String. E.g. 'n03995372'
            # Get corresponding integer class ID (1 to 1000)
            img_labels[i] = self._wnid_to_ilsvrc2012_id[wnid]

        return img_paths, img_labels

    def _load_test_set(self):
        """
        Loads training set image paths and corresponding labels into memory.
        :return: list of img paths (string), Empty list
        """
        img_paths = []
        return img_paths, None

    def _load_input_data(self):
        if self._portion == 'train':
            return self._load_training_set()
        elif self._portion == 'val':
            return self._load_validation_set()
        elif self._portion == 'test':
            return self._load_test_set()
        else:
            return None, None  # for deployment phase

    def _generator(self):
        """
        Runs over one epoch on the samples. Generate samples for training or inference.
        Used for getting samples from train/validation/test sets
        :return: Yields (one image path of type string, one one-hot 1000-d tensor of type tf.float32)
        """
        for i in range(len(self)):
            x = self._img_paths[i]
            # Optimized (convert to graph and batch) pre-processing for x is done outside this generator.

            # Test set does not have labels
            y = None if self._img_labels is None else self._img_labels[i]
            y = self._preprocess_y(y)  # Can be optimized by moving outside. But the gain would anyway be too little

            yield x, y

        return

    def _prepare_dataset(self):
        if self._portion is None:  # No samples to load during deployment phase
            return tf.data.Dataset()  # Empty placeholder

        with tf.device('/cpu:0'):
            # Images returned by the generator can have uneven resolution
            dataset = tf.data.Dataset.from_generator(self._generator,
                                                     output_types=(tf.dtypes.string, tf.dtypes.float32),
                                                     output_shapes=(tf.TensorShape([]),
                                                                    tf.TensorShape([1000])))
            # Prefetch samples using a separate thread for faster file-read
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            # Pre-process samples
            dataset = dataset.map(
                lambda x, y: (tf.py_function(func=self._preprocess_x_non_graph, inp=[x], Tout=tf.dtypes.float32), y),
                num_parallel_calls=4)

            # Batching should only be performed after the pre-processing has made all samples same shape
            dataset = dataset.batch(batch_size=self._batch_size, drop_remainder=False)

            # TODO: Now that all samples are same shape, possible to move this map() call outside tf.device('/cpu:0')?
            dataset = dataset.map(
                lambda x, y: (self._preprocess_x_with_graph(x), y),
                num_parallel_calls=4)

        return dataset

    # Don't use @tf.function for this. Reason: due to uneven image resolutions this function
    # cannot be converted to a graph
    def _preprocess_x_non_graph(self, x):
        """
        Preprocess image
        Contains portion of pre-processing that cannot be converted to a graph as the
        input image resolution is different across dataset
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
        Contains portion of pre-processing that can be converted to a graph due to
        predetermined input/output shapes across dataset

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
        Pre-process labels
        :param y: Either
            a tensor of shape (None, 1) containing integer class label (1, 1000) or
            None
        :return: A tensor of shape (None, 1000)
        """
        # Convert labels to one-hot
        if y is None:
            # No valid label available. Still return a tensor so that program doesn't crash
            y = tf.zeros(1000, dtype=tf.dtypes.float32)
        else:
            y = tf.one_hot(indices=y - 1, depth=1000, dtype=tf.dtypes.float32)

        return y
