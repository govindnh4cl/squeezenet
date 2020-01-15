from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, MaxPooling2D, \
    BatchNormalization, Activation, Dropout, AveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from my_logger import get_logger


class FireModule(tf.keras.Model):
    def __init__(self, squeeze_depth, expand_depth, data_format, batch_norm_decay, weight_decay):
        super().__init__(self)

        # Axis that represents channel in the feature map
        if data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        # --------------- Squeeze ----------------
        self.s = Conv2D(
            squeeze_depth,
            [1, 1],
            strides=1,
            activation='relu',
            data_format=data_format,
            padding='same',
            kernel_initializer='glorot_uniform')

        # --------------- Expand ----------------
        self.e1x1 = Conv2D(
            expand_depth,
            [1, 1],
            strides=1,
            activation='relu',
            data_format=data_format,
            padding='same',
            kernel_initializer='glorot_uniform')

        self.e3x3 = Conv2D(
            expand_depth,
            [3, 3],
            strides=1,
            activation='relu',
            data_format=data_format,
            padding='same',
            kernel_initializer='glorot_uniform')

        return

    def call(self, input_tensor, training):
        s_out = self.s(input_tensor)
        e_out_0 = self.e1x1(s_out)
        e_out_1 = self.e3x3(s_out)

        return tf.concat([e_out_0, e_out_1], self.channel_axis)


class Squeezenet(ABC, tf.keras.Model):
    """
    Base class for Squeezenet architectures tuned for different datasets (image resolution)
    """
    def __init__(self, cfg, name):
        tf.keras.Model.__init__(self)
        self.model_name = name
        self.logger = get_logger()
        self.logger.info('Using model: {:s}'.format(self.model_name))

        assert cfg.model.data_format in ('channels_first', 'channels_last')  # Sanity check
        self.data_format = cfg.model.data_format
        self._num_classes = cfg.dataset.num_classes
        self._weight_decay = cfg.model.weight_decay
        self._batch_norm_decay = cfg.model.batch_norm_decay
        self._input_shape = None
        self._training = False

        return

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, training):
        self._training = training


class Squeezenet_Imagenet(Squeezenet):
    """Original squeezenet architecture for 227x227 images."""
    def __init__(self, cfg):
        Squeezenet.__init__(self, cfg, 'squeezenet_imagenet')
        self._input_shape = (3, 227, 227)  # TODO: Can this be eliminated?
        num_classes = 1000

        # Axis that represents channel in the feature map
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.l_0 = Conv2D(96, [7, 7], strides=2, activation='relu', data_format=self.data_format, padding='same', kernel_initializer='glorot_uniform')
        self.l_1 = MaxPooling2D([3, 3], strides=2, data_format=self.data_format)

        self.l_2 = FireModule(16, 64, self.data_format, None, None)
        self.l_3 = FireModule(16, 64, self.data_format, None, None)
        self.l_4 = FireModule(32, 128, self.data_format, None, None)
        self.l_5 = MaxPooling2D([3, 3], strides=2, data_format=self.data_format)

        self.l_6 = FireModule(32, 128, self.data_format, None, None)
        self.l_7 = FireModule(48, 192, self.data_format, None, None)
        self.l_8 = FireModule(48, 192, self.data_format, None, None)
        self.l_9 = FireModule(64, 256, self.data_format, None, None)
        self.l_10 = MaxPooling2D([3, 3], strides=2, data_format=self.data_format)

        self.l_11 = FireModule(64, 256, self.data_format, None, None)
        self.l_12 = Dropout(rate=0.5)
        self.l_13 = Conv2D(num_classes, [1, 1], strides=1, activation='relu', data_format=self.data_format, padding='same', kernel_initializer=RandomNormal(mean=0., stddev=0.01))

        self.l_14 = GlobalAveragePooling2D(data_format=self.data_format)
        self.l_15 = Activation('softmax')

        return

    # TODO: This input signature is preventing us from switching to channels_first format
    @tf.function(input_signature=[tf.TensorSpec([None, 227, 227, 3], tf.float32)])
    def call(self, batch_x):
        x = batch_x

        x = self.l_0(x)
        x = self.l_1(x)

        x = self.l_2(x, None)
        x = self.l_3(x, None)
        x = self.l_4(x, None)
        x = self.l_5(x)

        x = self.l_6(x, None)
        x = self.l_7(x, None)
        x = self.l_8(x, None)
        x = self.l_9(x, None)
        x = self.l_10(x)

        x = self.l_11(x, None)
        x = self.l_12(x)
        x = self.l_13(x)

        x = self.l_14(x)
        if self.data_format == 'channels_first':
            logits = tf.squeeze(x, [2, 3], name='logits')
        else:
            logits = tf.squeeze(x, [1, 2], name='logits')
        out = self.l_15(logits)

        return out


class Squeezenet_CIFAR(Squeezenet):
    """Modified version of squeezenet for CIFAR images"""
    def __init__(self, cfg):
        Squeezenet.__init__(self, cfg, 'squeezenet_cifar')
        self._input_shape = (3, 32, 32)

        num_classes = 10

        # Axis that represents channel in the feature map
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3

        self.l_0 = Conv2D(96, [2, 2], activation='relu', kernel_regularizer=l2(self._weight_decay), data_format=self.data_format, padding='same')
        self.l_1 = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=self.channel_axis)
        self.l_2 = MaxPooling2D([2, 2], data_format=self.data_format)

        self.l_3 = FireModule(16, 64, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_4 = FireModule(16, 64, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_5 = FireModule(32, 128, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_6 = MaxPooling2D([2, 2], data_format=self.data_format)

        self.l_7 = FireModule(32, 128, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_8 = FireModule(48, 192, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_9 = FireModule(48, 192, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_10 = FireModule(64, 256, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_11 = MaxPooling2D([2, 2], data_format=self.data_format)

        self.l_12 = FireModule(64, 256, self.data_format, self._batch_norm_decay, self._weight_decay)
        self.l_13 = AveragePooling2D([4, 4], data_format=self.data_format)

        self.l_14 = Conv2D(num_classes, [1, 1], activation='relu', data_format=self.data_format, padding='same')
        self.l_15 = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=self.channel_axis)

        self.l_16 = Activation('softmax')
        return

    @tf.function(input_signature=[tf.TensorSpec([None, 32, 32, 3], tf.float32)])
    def call(self, batch_x):
        x = batch_x

        x = self.l_0(x)
        x = self.l_1(x, training=self.training)
        x = self.l_2(x)

        x = self.l_3(x, training=self.training)
        x = self.l_4(x, training=self.training)
        x = self.l_5(x, training=self.training)
        x = self.l_6(x)

        x = self.l_7(x, training=self.training)
        x = self.l_8(x, training=self.training)
        x = self.l_9(x, training=self.training)
        x = self.l_10(x, training=self.training)
        x = self.l_11(x)

        x = self.l_12(x, training=self.training)
        x = self.l_13(x)

        x = self.l_14(x)
        x = self.l_15(x, training=self.training)

        if self.data_format == 'channels_first':
            logits = tf.squeeze(x, [2, 3], name='logits')
        else:
            logits = tf.squeeze(x, [1, 2], name='logits')

        out = self.l_16(logits)

        return out

    def get_keras_model(self):
        inp = Input(shape=self._input_shape)
        model = Model(inputs=inp, outputs=self.call(inp, training=True))
        return model




