from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, \
    BatchNormalization, Activation, Dropout

from my_logger import get_logger


class FireModule(tf.keras.Model):
    def __init__(self, squeeze_depth, expand_depth):
        super().__init__(self)

        # Axis that represents channel in the feature map
        self.channel_axis = 3

        # --------------- Squeeze ----------------
        self.s = Conv2D(
            squeeze_depth,
            [1, 1],
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='glorot_uniform')

        self.bn_s = BatchNormalization()

        # --------------- Expand ----------------
        self.e1x1 = Conv2D(
            expand_depth,
            [1, 1],
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='glorot_uniform')

        # self.bn_e1x1 = BatchNormalization()

        self.e3x3 = Conv2D(
            expand_depth,
            [3, 3],
            strides=1,
            activation='relu',
            padding='same',
            kernel_initializer='glorot_uniform')

        # self.bn_e3x3 = BatchNormalization()

        self.bn_end = BatchNormalization()

        # TODO: bn_e1x1 and bn_e3x3 can be replaced with a single BN layer

        return

    def call(self, input_tensor, training):
        s_out = self.bn_s(self.s(input_tensor), training=training)

        e_out_0 = self.e1x1(s_out)
        e_out_1 = self.e3x3(s_out)
        e_out = tf.concat([e_out_0, e_out_1], self.channel_axis)

        out = self.bn_end(e_out)

        return out


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
        self._input_shape = None

        return


class Squeezenet_Imagenet(Squeezenet):
    """Original squeezenet architecture for imagenet """
    def __init__(self, cfg):
        Squeezenet.__init__(self, cfg, 'squeezenet_imagenet')
        num_classes = 1000

        # Axis that represents channel in the feature map
        self.channel_axis = 3

        self.l_0 = Conv2D(96, [7, 7], strides=2, activation='relu', padding='same')
        self.bn_l_0 = BatchNormalization()
        self.l_1 = MaxPooling2D([3, 3], strides=2)

        self.l_2 = FireModule(16, 64)
        self.l_3 = FireModule(16, 64)
        self.l_4 = FireModule(32, 128)
        self.l_5 = MaxPooling2D([3, 3], strides=2)

        self.l_6 = FireModule(32, 128)
        self.l_7 = FireModule(48, 192)
        self.l_8 = FireModule(48, 192)
        self.l_9 = FireModule(64, 256)
        self.l_10 = MaxPooling2D([3, 3], strides=2)

        self.l_11 = FireModule(64, 256)
        self.l_12 = Dropout(rate=0.5)
        self.l_13 = Conv2D(num_classes, [1, 1], strides=1, activation='relu', padding='same')
        self.bn_l_13 = BatchNormalization()

        self.l_14 = GlobalAveragePooling2D()
        self.l_15 = Activation('softmax')

        return

    @tf.function
    def call(self, batch_x, training):
        x = batch_x

        x = self.l_0(x)
        x = self.bn_l_0(x, training=training)
        x = self.l_1(x)

        x = self.l_2(x, training=training)
        x = self.l_3(x, training=training)
        x = self.l_4(x, training=training)
        x = self.l_5(x)

        x = self.l_6(x, training=training)
        x = self.l_7(x, training=training)
        x = self.l_8(x, training=training)
        x = self.l_9(x, training=training)
        x = self.l_10(x)

        x = self.l_11(x, training=training)
        x = self.l_12(x, training=training)
        x = self.l_13(x)
        x = self.bn_l_13(x, training=training)

        x = self.l_14(x)
        out = self.l_15(x)

        return out



