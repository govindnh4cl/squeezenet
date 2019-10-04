from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from my_logger import get_logger


class FireModule(tf.keras.Model):
    def __init__(self, squeeze_depth, expand_depth, batch_norm_decay, weight_decay):
        super().__init__(self)

        # --------------- Squeeze ----------------
        self.s_0 = Conv2D(
            squeeze_depth,
            [1, 1],
            strides=1,
            activation='relu',
            kernel_regularizer=l2(weight_decay),
            data_format='channels_first',
            padding='same')

        self.s_1 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)

        # --------------- Expand ----------------
        self.e1x1_0 = Conv2D(
            expand_depth,
            [1, 1],
            strides=1,
            activation='relu',
            kernel_regularizer=l2(weight_decay),
            data_format='channels_first',
            padding='same')

        self.e1x1_1 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)

        self.e3x3_0 = Conv2D(
            expand_depth,
            [3, 3],
            activation='relu',
            kernel_regularizer=l2(weight_decay),
            data_format='channels_first',
            padding='same')

        self.e3x3_1 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)
        return

    def call(self, input_tensor, training=False):
        s_out = self.s_0(input_tensor)
        s_out = self.s_1(s_out, training=training)

        e_out_0 = self.e1x1_0(s_out)
        e_out_0 = self.e1x1_1(e_out_0, training=training)

        e_out_1 = self.e3x3_0(s_out)
        e_out_1 = self.e3x3_1(e_out_1, training=training)

        return tf.concat([e_out_0, e_out_1], 1)


class Squeezenet(ABC, tf.keras.Model):
    """
    Base class for Squeezenet architectures tuned for different datasets (image resolution)
    """
    def __init__(self, cfg, name):
        tf.keras.Model.__init__(self)
        self.model_name = name
        self.logger = get_logger()
        self.logger.info('Using model: {:s}'.format(self.model_name))

        self._num_classes = cfg.dataset.num_classes
        self._weight_decay = cfg.model.weight_decay
        self._batch_norm_decay = cfg.model.batch_norm_decay
        self._input_shape = None

        return


class Squeezenet_Imagenet(Squeezenet):
    """Original squeezenet architecture for 224x224 images."""
    def __init__(self, cfg):
        Squeezenet.__init__(self, cfg, 'squeezenet_imagenet')
        self._input_shape = (3, 224, 224)

    def _define(self, num_classes=1000):
        inp = Input(shape=self._input_shape)

        net = Conv2D(96, [7, 7], strides=2, activation = 'relu', kernel_regularizer=l2(self._weight_decay), data_format='channels_first', padding='same')(inp)

        net = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)(net)
        net = MaxPooling2D([3, 3], strides=2, data_format='channels_first')(net)
        net = fire_module(net, 16, 64, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 16, 64, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 32, 128, self._batch_norm_decay, self._weight_decay)
        net = MaxPooling2D([3, 3], strides=2, data_format='channels_first')(net)
        net = fire_module(net, 32, 128, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 48, 192, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 48, 192, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 64, 256, self._batch_norm_decay, self._weight_decay)
        net = MaxPooling2D([3, 3], strides=2, data_format='channels_first')(net)
        net = fire_module(net, 64, 256, self._batch_norm_decay, self._weight_decay)
        net = Conv2D(num_classes, [1, 1], strides=1, activation = 'relu', data_format='channels_first', padding='same')(net)
        net = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)(net)
        net = AveragePooling2D([13, 13], strides=1, data_format='channels_first')(net)
        logits = tf.squeeze(net, [2], name='logits')
        out = Activation('softmax')(logits)

        model = Model(inputs=inp, outputs=out)
        return model


class Squeezenet_CIFAR(Squeezenet):
    """Modified version of squeezenet for CIFAR images"""
    def __init__(self, cfg):
        Squeezenet.__init__(self, cfg, 'squeezenet_cifar')
        self._input_shape = (3, 32, 32)

        num_classes = 10
        self.l_0 = Conv2D(96, [2, 2], activation='relu', kernel_regularizer=l2(self._weight_decay), data_format='channels_first', padding='same')
        self.l_1 = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)
        self.l_2 = MaxPooling2D([2, 2], data_format='channels_first')

        self.l_3 = FireModule(16, 64, self._batch_norm_decay, self._weight_decay)
        self.l_4 = FireModule(16, 64, self._batch_norm_decay, self._weight_decay)
        self.l_5 = FireModule(32, 128, self._batch_norm_decay, self._weight_decay)
        self.l_6 = MaxPooling2D([2, 2], data_format='channels_first')

        self.l_7 = FireModule(32, 128, self._batch_norm_decay, self._weight_decay)
        self.l_8 = FireModule(48, 192, self._batch_norm_decay, self._weight_decay)
        self.l_9 = FireModule(48, 192, self._batch_norm_decay, self._weight_decay)
        self.l_10 = FireModule(64, 256, self._batch_norm_decay, self._weight_decay)
        self.l_11 = MaxPooling2D([2, 2], data_format='channels_first')

        self.l_12 = FireModule(64, 256, self._batch_norm_decay, self._weight_decay)
        self.l_13 = AveragePooling2D([4, 4], data_format='channels_first')

        self.l_14 = Conv2D(num_classes, [1, 1], activation='relu', data_format='channels_first', padding='same')
        self.l_15 = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)

        self.l_16 = Activation('softmax')
        return

    @tf.function
    def call(self, batch_x, training=False):
        x = batch_x

        x = self.l_0(x)
        x = self.l_1(x, training=training)
        x = self.l_2(x)

        x = self.l_3(x, training=training)
        x = self.l_4(x, training=training)
        x = self.l_5(x, training=training)
        x = self.l_6(x)

        x = self.l_7(x, training=training)
        x = self.l_8(x, training=training)
        x = self.l_9(x, training=training)
        x = self.l_10(x, training=training)
        x = self.l_11(x)

        x = self.l_12(x, training=training)
        x = self.l_13(x)

        x = self.l_14(x)
        x = self.l_15(x, training=training)
        logits = tf.squeeze(x, [2, 3], name='logits')
        out = self.l_16(logits)

        return out

    def get_keras_model(self):
        inp = Input(shape=self._input_shape)
        model = Model(inputs=inp, outputs=self.call(inp, training=True))
        return model




