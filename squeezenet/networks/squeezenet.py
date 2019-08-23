from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                batch_norm_decay,
                weight_decay):
    net = _squeeze(inputs, squeeze_depth, batch_norm_decay, weight_decay)
    net = _expand(net, expand_depth, batch_norm_decay, weight_decay)
    return net


def _squeeze(inputs, num_outputs, batch_norm_decay, weight_decay):
    x = Conv2D(num_outputs,
               [1, 1],
               strides=1,
               activation='relu',
               kernel_regularizer=l2(weight_decay),
               data_format='channels_first',
               padding='same')(inputs)

    x = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)(x)

    return x


def _expand(inputs, num_outputs, batch_norm_decay, weight_decay):
    e1x1 = Conv2D(num_outputs,
                  [1, 1],
                  strides=1,
                  activation = 'relu',
                  kernel_regularizer=l2(weight_decay),
                  data_format='channels_first',
                  padding='same')(inputs)

    e1x1 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)(e1x1)

    e3x3 = Conv2D(num_outputs,
                  [3, 3],
                  activation='relu',
                  kernel_regularizer=l2(weight_decay),
                  data_format='channels_first',
                  padding='same')(inputs)

    e3x3 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)(e3x3)

    return tf.concat([e1x1, e3x3], 1)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._input_shape = (3, 224, 224)

    def build(self):
        return self._squeezenet(self._num_classes)

    def _squeezenet(self, num_classes=1000):
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


class Squeezenet_CIFAR(object):
    """Modified version of squeezenet for CIFAR images"""
    name = 'squeezenet_cifar'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._input_shape = (3, 32, 32)

    def build(self):
        return self._squeezenet_cifar(self._num_classes)

    def _squeezenet_cifar(self, num_classes=10):
        inp = Input(shape=self._input_shape)

        net = Conv2D(96, [2, 2], activation='relu', kernel_regularizer=l2(self._weight_decay), data_format='channels_first', padding='same')(inp)
        net = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)(net)
        net = MaxPooling2D([2, 2], data_format='channels_first')(net)
        net = fire_module(net, 16, 64, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 16, 64, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 32, 128, self._batch_norm_decay, self._weight_decay)
        net = MaxPooling2D([2, 2], data_format='channels_first')(net)
        net = fire_module(net, 32, 128, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 48, 192, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 48, 192, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 64, 256, self._batch_norm_decay, self._weight_decay)
        net = MaxPooling2D([2, 2], data_format='channels_first')(net)
        net = fire_module(net, 64, 256, self._batch_norm_decay, self._weight_decay)
        net = AveragePooling2D([4, 4], data_format='channels_first')(net)
        net = Conv2D(num_classes, [1, 1], activation='relu', data_format='channels_first', padding='same')(net)
        net = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)(net)
        logits = tf.squeeze(net, [2, 3], name='logits')
        out = Activation('softmax')(logits)

        model = Model(inputs=inp, outputs=out)
        return model


