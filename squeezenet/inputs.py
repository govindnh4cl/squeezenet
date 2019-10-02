import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from tensorflow import data

from my_logger import get_logger


class Pipeline(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger()
        self._setup_dataset_objects()
        return

    def _setup_dataset_objects(self):
        """
        Sets up Tensorflow dataset objects
        :return: None. Creates objects within class
        """
        if 'gpu' in self.cfg.hardware.device:
            self.logger.warning('Holding all dataset operations on CPU as I '
                                'am not able set them on GPU at present due to unknown reasons')

        with tf.device('/cpu:0'):  # This forces dataset operation on CPU
            # Load all images from CIFAR10
            self.logger.info('Getting input data... ')
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            self.logger.info('Got all input data.')

            if (self.cfg.dataset.train.enable or self.cfg.dataset.val.enable) is True:
                x_train = tf.cast(x_train, tf.float32)
                # TODO: Apply pre-processing
                # Convert to channel-first
                x_train = tf.transpose(x_train, [0, 3, 1, 2])
                # Convert labels to one-hot
                y_train = tf.one_hot(indices=y_train, depth=10, axis=1, on_value=1, off_value=0, dtype=tf.int32)

                # Change shape from (batch_size, num_classes, 1) to (batch_size, num_classes)
                y_train = tf.squeeze(y_train, axis=[2])

                count_train_val = len(x_train)
                self.count_train = int(count_train_val * 0.9)
                self.count_val = count_train_val - self.count_train
                self.train_val_datset = tf.data.Dataset.from_tensor_slices((x_train, y_train))  # Init from data
                self.train_val_datset = self.train_val_datset.shuffle(buffer_size=count_train_val)  # Shuffle

                if self.cfg.dataset.train.enable is True:
                    self.train_dataset = self.train_val_datset.take(self.count_train)  # Get just the train set

                    # Convert image dtype to float32
                    # TODO: I think this is unnecessary
                    self.train_dataset = self.train_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
                    # Convert into batched datasets
                    self.train_dataset = self.train_dataset.batch(self.cfg.dataset.train.batch_size, drop_remainder=False)
                    self.logger.info('Training batch size: {:d} \tCount steps per epoch: {:d}'.format(
                        self.cfg.dataset.train.batch_size, np.round(self.count_train / self.cfg.dataset.train.batch_size).astype(int)))
                else:
                    self.train_dataset = None
                    self.count_train = 0

                if self.cfg.dataset.val.enable is True:
                    self.val_dataset = self.train_val_datset.skip(self.count_train)  # Get just the validation set

                    # Convert image dtype to float32
                    self.val_dataset = self.val_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
                    # Convert into batched datasets
                    self.val_dataset = self.val_dataset.batch(self.cfg.dataset.val.batch_size, drop_remainder=False)
                    self.logger.info('Validation batch size: {:d} \tCount steps per epoch: {:d}'.format(
                        self.cfg.dataset.train.batch_size, np.round(self.count_train/self.cfg.dataset.train.batch_size).astype(int)))
                else:
                    self.val_dataset = None
                    self.count_val = 0

            if self.cfg.dataset.test.enable is True:
                self.cfg.dataset.test.batch_size = self.cfg.dataset.test.batch_size
                x_test = tf.cast(x_test, tf.float32)
                # Convert to channel-first
                x_test = tf.transpose(x_test, [0, 3, 1, 2])
                # Convert labels to one-hot
                y_test = tf.one_hot(indices=y_test, depth=10, axis=1, on_value=1, off_value=0, dtype=tf.int32)

                # Change shape from (batch_size, num_classes, 1) to (batch_size, num_classes)
                y_test = tf.squeeze(y_test, axis=[2])

                self.count_test = len(x_test)
                self.test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))  # Init from data
                self.test_dataset = self.test_dataset.shuffle(buffer_size=self.count_test)  # Shuffle
                # Convert image dtype to float32
                self.test_dataset = self.test_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
                # Convert into batched datasets
                self.test_dataset = self.test_dataset.batch(self.cfg.dataset.test.batch_size, drop_remainder=False)
                self.logger.info('Test batch size: {:d} \tCount steps per epoch: {:d}'.format(
                    self.cfg.dataset.test.batch_size, np.round(self.count_test / self.cfg.dataset.test.batch_size).astype(int)))
            else:  # No test phase
                self.test_dataset = None
                self.count_test = 0

        return

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset


class _InputProcessor(object):
    def __init__(self,
                 batch_size,
                 num_threads,
                 repeat,
                 shuffle,
                 # shuffle_buffer,
                 seed,
                 distort_image=None,
                 target_image_size=None):
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.repeat = repeat
        self.shuffle = shuffle
        # self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.distort_image = distort_image
        self.target_image_size = target_image_size

    def from_tfrecords(self, files):
        dataset = data.TFRecordDataset(files)
        dataset = dataset.map(
            map_func=self._preprocess_example,
            num_parallel_calls=self.num_threads
        )
        dataset = dataset.repeat(self.repeat)
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=self.seed
            )
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _preprocess_example(self, serialized_example):
        parsed_example = self._parse_serialized_example(serialized_example)
        image = self._preprocess_image(parsed_example['image'])
        return {'image': image}, parsed_example['label']

    def _preprocess_image(self, raw_image):
        image = tf.image.decode_jpeg(raw_image, channels=3)
        image = tf.image.resize_images(image, self.target_image_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.distort_image:
            image = tf.image.random_flip_left_right(image)
        image = tf.transpose(image, [2, 0, 1])
        return image

    @staticmethod
    def _parse_serialized_example(serialized_example):
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }
        return tf.parse_single_example(serialized=serialized_example,
                                       features=features)
