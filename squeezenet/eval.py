import tensorflow as tf


def get_categorical_accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    evaluator = tf.metrics.CategoricalAccuracy()

    evaluator.update_state(y_true, y_pred)
    acc = evaluator.result()
    return acc

