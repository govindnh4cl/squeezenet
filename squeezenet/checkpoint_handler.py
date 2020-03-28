import tensorflow as tf

from my_logger import get_logger


class CheckpointHandler:
    """
    Handles checkpoint related tasks
    """
    def __init__(self, cfg):
        self._logger = get_logger()
        self._cfg = cfg

        # A tensor to hold the checkpoint index
        # Needed just for knowing what epoch a checkpoint belongs to
        self._ckpt_counter = None

        self._ckpt = None  # An object of class tf.train.Checkpoint()
        self._ckpt_mngr = None  # An object of class tf.train.CheckpointManager()
        self._ckpt_status = None  # Holds result of self._ckpt.restore(). Used to verify a successful restore

        return

    def load_checkpoint(self, net, opt):
        """
        Loads checkpoint from directory.
        :param net: Network to be stored
        :param opt: Optimizer (and its states) to be stored
        :return: None
        """
        # Checkpoint restoration
        self._ckpt_counter = tf.Variable(initial_value=-1, trainable=False, dtype=tf.int64)  # Stores epoch ID
        self._ckpt = tf.train.Checkpoint(model=net, ckpt_counter=self._ckpt_counter, optimizer=opt)

        self._ckpt_mngr = tf.train.CheckpointManager(checkpoint=self._ckpt,
                                                     directory=self._cfg.directories.dir_ckpt_train,
                                                     max_to_keep=self._cfg.train.keep_n_checkpoints)

        if self._ckpt_mngr.latest_checkpoint:  # Search for a checkpoint in the checkpoint directory
            self._ckpt_status = self._ckpt.restore(self._ckpt_mngr.latest_checkpoint)  # Restore
            self._logger.info('Training checkpoint: Restored from: {:s}'.format(self._ckpt_mngr.latest_checkpoint))
            self._logger.info('Training checkpoint: Checkpoint counter: {:d}'.format(self._ckpt_counter.numpy()))
        else:
            self._logger.info('Training checkpoint: Not found. Init weights from scratch.')

        return

    def verify_checkpoint_restore(self):
        """
        Confirms that a checkpoint was loader properly.
        In graph mode, this can be called right after the self._ckpt.restore() call.
        In eager mode this should be called at least after one batch has been processed.
        :return: None
        """
        if self._ckpt_status is not None:
            self._ckpt_status.assert_consumed()  # Verify the correctness of checkpoint loading

        return

    def save_checkpoint(self):
        """
        Saves a new checkpoint to the disk
        :return:
        """
        assert self._ckpt is not None  # Confirm that save only be called after self.load_checkpoint()

        self._ckpt.ckpt_counter.assign_add(1)  # Increment checkpoint id
        if int(self._ckpt.ckpt_counter) % self._cfg.train.checkpoint_interval == 0:
            save_path = self._ckpt_mngr.save(checkpoint_number=int(self._ckpt.ckpt_counter))  # Save checkpoint
            self._logger.info('Saved checkpoint at {:s}'.format(save_path))
        return



