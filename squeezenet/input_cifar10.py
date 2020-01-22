import numpy as np
import tensorflow as tf

from squeezenet.inputs import Pipeline


class PipelineCIFAR10(Pipeline):
    def __init__(self, cfg):
        Pipeline.__init__(self, cfg)

        if self._cfg.preprocessing.enable_augmentation is True:
            # Some data augmentation calls only support 'channels_last' format
            assert self._cfg.model.data_format == 'channels_last'

        self._setup_dataset_objects()
        return

    def _setup_dataset_objects(self):
        """
        Sets up Tensorflow dataset objects
        :return: None. Creates objects within class
        """
        if 'gpu' in self._cfg.hardware.device:
            self._logger.warning('Holding all dataset operations on CPU as I '
                                'am not able set them on GPU at present due to unknown reasons')

        with tf.device('/cpu:0'):  # This forces dataset operation on CPU
            # Load all images from CIFAR10
            self._logger.info('Getting input data... ')
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            self._logger.info('Got all input data.')

            if (self._cfg.dataset.train.enable or self._cfg.dataset.val.enable) is True:
                x_train = tf.cast(x_train, tf.float32)

                # Convert to channel-first
                if self._cfg.model.data_format == 'channels_first':
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

                if self._cfg.dataset.train.enable is True:
                    self.train_dataset = self.train_val_datset.take(self.count_train)  # Get just the train set

                    # Convert image dtype to float32
                    # TODO: I think this is unnecessary
                    self.train_dataset = self.train_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
                    # Convert into batched datasets
                    self.train_dataset = self.train_dataset.batch(self._cfg.dataset.train.batch_size, drop_remainder=False)
                    # Image normalization
                    self.train_dataset = self.train_dataset.map(lambda x, y: (self.normalize_image(x), y))

                    if self._cfg.preprocessing.enable_augmentation is True:  # Data augmentation
                        self.train_dataset = self.train_dataset.map(lambda x, y: (self._perturb_image(x), y))

                    self._logger.info('Training batch size: {:d} \tCount steps per epoch: {:d}'.format(
                        self._cfg.dataset.train.batch_size, np.round(self.count_train / self._cfg.dataset.train.batch_size).astype(int)))
                else:
                    self.train_dataset = None
                    self.count_train = 0

                if self._cfg.dataset.val.enable is True:
                    self.val_dataset = self.train_val_datset.skip(self.count_train)  # Get just the validation set

                    # Convert image dtype to float32
                    # TODO: I think this is unnecessary
                    self.val_dataset = self.val_dataset.map(lambda x, y: (tf.cast(x, tf.float32), y))
                    # Convert into batched datasets
                    self.val_dataset = self.val_dataset.batch(self._cfg.dataset.val.batch_size, drop_remainder=False)
                    # Image normalization
                    self.val_dataset = self.val_dataset.map(lambda x, y: (self.normalize_image(x), y))
                    self._logger.info('Validation batch size: {:d} \tCount steps per epoch: {:d}'.format(
                        self._cfg.dataset.val.batch_size, np.round(self.count_val/self._cfg.dataset.val.batch_size).astype(int)))
                else:
                    self.val_dataset = None
                    self.count_val = 0

            if self._cfg.dataset.test.enable is True:
                self._cfg.dataset.test.batch_size = self._cfg.dataset.test.batch_size
                # Convert image dtype to float32
                # TODO: I think this is unnecessary
                x_test = tf.cast(x_test, tf.float32)

                # Convert to channel-first
                if self._cfg.model.data_format == 'channels_first':
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
                self.test_dataset = self.test_dataset.batch(self._cfg.dataset.test.batch_size, drop_remainder=False)
                # Image normalization
                self.test_dataset = self.test_dataset.map(lambda x, y: (self.normalize_image(x), y))
                self._logger.info('Test batch size: {:d} \tCount steps per epoch: {:d}'.format(
                    self._cfg.dataset.test.batch_size, np.round(self.count_test / self._cfg.dataset.test.batch_size).astype(int)))
            else:  # No test phase
                self.test_dataset = None
                self.count_test = 0

        return

    @staticmethod
    def normalize_image(img_batch):
        """
        Applies normalization on a batch of images
        :param img_batch: Image batch of shape (None, 3, ht, wd). Channel order is BGR
        :return: Normalized image of same shape (as input) and dtype tf.float32
        """
        # TODO: The n/w may also accept a single image (shape = 3). So remove this assert below
        assert len(img_batch.shape) == 4  # Sanity check.

        # TODO: Normalize according to the original paper
        return tf.image.per_image_standardization(img_batch)

    def _perturb_image(self, img_batch):
        """
        Aplies a random perturbation for data-augmentation during training
        :param img_batch: Image batch of shape (None, 3, ht, wd). Channel order is BGR
        :return:
        """
        img_batch = tf.image.random_flip_left_right(img_batch, seed=1337)
        return img_batch

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
