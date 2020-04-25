from collections import OrderedDict


class CustomLearningRateScheduler:
    def __init__(self):
        self._lr_schedule = OrderedDict()

        self._lr_schedule[22] = 0.04  # Use this learning rate for epoch index 0 to 21 (inclusive)
        self._lr_schedule[27] = 0.001  # Use this learning rate for epoch index 22 to 26 (inclusive)
        self._lr_schedule[42] = 0.0002
        self._lr_schedule[72] = 0.0001
        self._lr_schedule[float('inf')] = 0.00001

        pass

    def get_learning_rate(self, epoch_idx):
        """
        Returns a learning rate based on the epoch index
        :param epoch_idx: Epoch index. 0 to infinity
        :return:
        """
        for i in self._lr_schedule:
            if epoch_idx < i:
                lr = self._lr_schedule[i]
                return lr

        assert False  # Should never come here


