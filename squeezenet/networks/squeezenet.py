from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python as tf
from tensorflow.python.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.regularizers import l2


def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                batch_norm_decay,
                weight_decay):
    net = _squeeze(inputs, squeeze_depth, batch_norm_decay, weight_decay)
    net = _expand(net, expand_depth, batch_norm_decay, weight_decay)
    return net


def _squeeze(inputs, num_outputs, batch_norm_decay, weight_decay):
    x = Conv2D(num_outputs, [1, 1], stride=1, kernel_regularizer=l2(weight_decay), data_format='channels_first')(inputs)
    x = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)(x)
    return x


def _expand(inputs, num_outputs, batch_norm_decay, weight_decay):
    e1x1 = Conv2D(num_outputs, [1, 1], stride=1, kernel_regularizer=l2(weight_decay), data_format='channels_first')(inputs)
    e1x1 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)(e1x1)

    e3x3 = Conv2D(num_outputs, [3, 3], kernel_regularizer=l2(weight_decay), data_format='channels_first')(inputs)
    e3x3 = BatchNormalization(momentum=batch_norm_decay, fused=True, axis=1)(e3x3)

    return tf.concat([e1x1, e3x3], 1)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        return self._squeezenet(x, self._num_classes)

    def _squeezenet(self, images, num_classes=1000):
        net = Conv2D(96, [7, 7], strides=2, kernel_regularizer=l2(self._weight_decay), data_format='channels_first')(images)
        net = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)(net)
        net = MaxPooling2D([3, 3], strides=2, data_format='channels_first')(net)
        net = fire_module(net, 16, 64, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 16, 64, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 32, 128, self._batch_norm_decay, self._weight_decay)
        net = MaxPooling2D([3, 3], stride=2, data_format='channels_first')(net)
        net = fire_module(net, 32, 128, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 48, 192, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 48, 192, self._batch_norm_decay, self._weight_decay)
        net = fire_module(net, 64, 256, self._batch_norm_decay, self._weight_decay)
        net = MaxPooling2D([3, 3], stride=2, data_format='channels_first')(net)
        net = fire_module(net, 64, 256, self._batch_norm_decay, self._weight_decay)
        net = Conv2D(num_classes, [1, 1], stride=1, data_format='channels_first')(net)
        net = BatchNormalization(momentum=self._batch_norm_decay, fused=True, axis=1)(net)
        net = AveragePooling2D([13, 13], stride=1, data_format='channels_first')(net)
        logits = tf.squeeze(net, [2], name='logits')
        return logits


class Squeezenet_CIFAR(object):
    """Modified version of squeezenet for CIFAR images"""
    name = 'squeezenet_cifar'

    def __init__(self, args):
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        return self._squeezenet(x)

    @staticmethod
    def _squeezenet(images, num_classes=10):
        # TODO: Pass the default values of self._weight_decay and self._batch_norm_decay
        net = Conv2D(images, 96, [2, 2])

        net = MaxPooling2D(net, [2, 2])
        net = fire_module(net, 16, 64)
        net = fire_module(net, 16, 64)
        net = fire_module(net, 32, 128)
        net = MaxPooling2D(net, [2, 2])
        net = fire_module(net, 32, 128)
        net = fire_module(net, 48, 192)
        net = fire_module(net, 48, 192)
        net = fire_module(net, 64, 256)
        net = MaxPooling2D(net, [2, 2])
        net = fire_module(net, 64, 256)
        net = AveragePooling2D(net, [4, 4])
        net = Conv2D(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
        logits = tf.squeeze(net, [2], name='logits')
        return logits


