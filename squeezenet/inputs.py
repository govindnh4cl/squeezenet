import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from tensorflow import data


logger = tf.get_logger()


class Pipeline(object):
    def __init__(self, args):
        self.batch_size = args.batch_size

        # Load all images from CIFAR10
        print('Getting input data... ')
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            keras.datasets.cifar10.load_data()
        print('Got all input data.')

        # TODO: Apply pre-processing
        self.x_train = tf.cast(self.x_train, tf.float32)

        # Convert to channel-first
        self.x_train = tf.transpose(self.x_train, [0, 3, 1, 2])
        self.x_test = tf.transpose(self.x_test, [0, 3, 1, 2])

        # Convert labels to one-hot
        self.y_train = tf.one_hot(indices=self.y_train, depth=10, axis=1, on_value=1, off_value=0, dtype=tf.int32)
        self.y_test = tf.one_hot(indices=self.y_test, depth=10, axis=1, on_value=1, off_value=0, dtype=tf.int32)

        # Change shape from (batch_size, num_classes, 1) to (batch_size, num_classes)
        self.y_train = tf.squeeze(self.y_train, axis=[2])
        self.y_test = tf.squeeze(self.y_test, axis=[2])

        count_train_val = len(self.x_train)
        self.count_test = len(self.x_test)
        self.count_train = int(count_train_val * 0.9)
        self.count_val = count_train_val - self.count_train
        self.count_total = self.count_train + self.count_val + self.count_test
        self.train_val_datset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))  # Init from data
        self.train_val_datset = self.train_val_datset.shuffle(buffer_size=count_train_val)  # Shuffle
        self.train_dataset = self.train_val_datset.take(self.count_train)  # Get just the train set
        self.val_dataset = self.train_val_datset.skip(self.count_train)  # Get just the validation set
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))  # Init from data
        self.test_dataset = self.test_dataset.shuffle(buffer_size=self.count_test)  # Shuffle

        # Convert image dtype to float32
        self.train_dataset = self.train_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
        self.val_dataset = self.val_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
        self.test_dataset = self.test_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))

        # Convert into batched datasets
        self.train_dataset = self.train_dataset.batch(self.batch_size, drop_remainder=False)
        self.val_dataset = self.val_dataset.batch(self.batch_size, drop_remainder=False)
        self.test_dataset = self.test_dataset.batch(self.batch_size, drop_remainder=False)

        print('Count samples: Train={:d}({:.1f}%) Validation={:d}({:.1f}%) Test={:d}({:.1f}%)'.format(
            self.count_train, 100 * self.count_train / self.count_total,
            self.count_val, 100 * self.count_val / self.count_total,
            self.count_test, 100 * self.count_test / self.count_total))

        print('Batch size: {:d} Steps: Train={:d} Validation={:d} Test={:d}'.format(
              self.batch_size,
              np.round(self.count_train/self.batch_size).astype(int),
              np.round(self.count_val/self.batch_size).astype(int),
              np.round(self.count_test/self.batch_size).astype(int)))

        # TODO: For cifar10, we don't need to resize, but for imagenet we might have to apply preprocessing
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
