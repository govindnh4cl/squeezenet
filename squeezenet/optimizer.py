from collections import OrderedDict


class CustomLearningRateScheduler:
    def __init__(self):
        self._lr_schedule = OrderedDict()

        self._lr_schedule[30] = 0.04  # Use this learning rate for epoch index 0 to 24 (inclusive)
        self._lr_schedule[40] = 0.02  # Use this learning rate for epoch index 10 to 19 (inclusive)
        self._lr_schedule[45] = 0.01
        self._lr_schedule[50] = 0.005000
        self._lr_schedule[55] = 0.002500
        self._lr_schedule[60] = 0.001000
        self._lr_schedule[65] = 0.000500
        self._lr_schedule[70] = 0.000250
        self._lr_schedule[75] = 0.000100
        self._lr_schedule[80] = 0.000050
        self._lr_schedule[85] = 0.000025
        # self._lr_schedule[90] = 0.000010
        # self._lr_schedule[95] = 0.000005
        # self._lr_schedule[100] = 0.0000025
        self._lr_schedule[float('inf')] = 0.000001

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


