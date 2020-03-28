import tensorflow as tf


def get_accuracy(y_true_label, y_pred):
    """

    :param y_true_label: 1-D array of integers representing class labels (0 to 999)
    :param y_pred: 2-D array of shape (num_samples, num_classes) with scores of each class. Higher is better.
    :return: Two floats each ranging from 0 to 1. Top-1 and Top-5 accuracy
    """
    count_samples = len(y_true_label)
    assert count_samples == len(y_pred)

    top1_acc = tf.math.reduce_sum(tf.cast(tf.math.in_top_k(y_true_label, y_pred, 1), tf.float32)) / count_samples
    top5_acc = tf.math.reduce_sum(tf.cast(tf.math.in_top_k(y_true_label, y_pred, 5), tf.float32)) / count_samples
    return top1_acc, top5_acc

