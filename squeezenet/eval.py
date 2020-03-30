import tensorflow as tf


def get_accuracy(y_true, y_pred):
    """

    :param y_true: 2-D array of shape (num_samples, num_classes). Each row is one-hot vector
    :param y_pred: 2-D array of shape (num_samples, num_classes) with scores of each class. Higher is better.
    :return: Two floats each ranging from 0 to 1. Top-1 and Top-5 accuracy
    """
    count_samples = len(y_true)
    assert count_samples == len(y_pred)

    y_true_label = tf.math.argmax(y_true, axis=1)  # 1-D tensor of integer labels. Values are 0-999
    y_true_label = tf.cast(y_true_label, dtype=tf.int32)

    sorted_pred_idx = tf.argsort(y_pred, axis=1, stable=True, direction='DESCENDING')

    # Boolean 1-D tensors
    assert y_true_label.shape == sorted_pred_idx[:, 0].shape  # Sanity check
    correct_top1 = tf.equal(sorted_pred_idx[:, 0], y_true_label)

    y_true_label = tf.expand_dims(y_true_label, axis=-1)
    correct_top5 = tf.reduce_any(tf.equal(sorted_pred_idx[:, :5], y_true_label), axis=1)

    count_correct_top1 = tf.reduce_sum(tf.cast(correct_top1, dtype=tf.int32))
    count_correct_top5 = tf.reduce_sum(tf.cast(correct_top5, dtype=tf.int32))

    return count_correct_top1/count_samples, count_correct_top5/count_samples

