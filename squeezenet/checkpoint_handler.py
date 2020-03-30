import tensorflow as tf

from my_logger import get_logger


class CheckpointHandler:
    """
    Handles checkpoint related tasks
    """
    def __init__(self, cfg):
        """

        :param cfg: Configuration
        """
        self._logger = get_logger()
        self._cfg = cfg

        # A tensor to hold the checkpoint integer id
        self._ckpt_id = None

        self._ckpt = None  # An object of class tf.train.Checkpoint()
        self._ckpt_mngr = None  # An object of class tf.train.CheckpointManager()
        self._ckpt_status = None  # Holds result of self._ckpt.restore(). Used to verify a successful restore

        return

    def load_checkpoint(self, to_store, ckpt2load='latest'):
        """
        Loads checkpoint from directory.
        :param to_store: A dictionary of Python object stored in checkpoints
        :param ckpt2load: Specify which checkpoint to load. Supported values
            'latest': Use the most recent checkpoint
            'scratch': Don't load from any checkpoint
            <int>: Specify the integer ID of the checkpoint to load from
        :return: None
        """
        self._ckpt_id = tf.Variable(initial_value=-1, trainable=False, dtype=tf.int64)  # Stores epoch ID
        self._ckpt = tf.train.Checkpoint(model=to_store['net'], ckpt_counter=self._ckpt_id, optimizer=to_store['opt'])

        self._ckpt_mngr = tf.train.CheckpointManager(checkpoint=self._ckpt,
                                                     directory=self._cfg.directories.dir_ckpt_train,
                                                     max_to_keep=self._cfg.train.keep_n_checkpoints)

        if ckpt2load == 'latest':
            ckpt_path = self._ckpt_mngr.latest_checkpoint
            if ckpt_path is None:
                self._logger.info('No existing checkpoint found.')
            else:
                ckpt_id = int(ckpt_path.split('-')[-1])

        elif ckpt2load == 'scratch':
            # Nothing needs to be done as networks weights will be randomly initialized
            ckpt_path = None

        else:
            ckpt_id = int(ckpt2load)
            ckpt_ids = [int(x.split('-'[-1])) for x in self._ckpt_mngr.checkpoints]

            try:
                ckpt_path = self._ckpt_mngr.checkpoints[ckpt_ids.index(ckpt_id)]
            except ValueError:
                self._logger.error("Couldn't find checkpoint for ID: {:d}. Available checkpoint IDs: {:}"
                                   .format(ckpt_id, ckpt_ids))

                ckpt_path = None  # Continue without loading checkpoint

        if ckpt_path is None:
            # Nothing needs to be done
            self._logger.info('Not restoring checkpoint. Init weights from scratch.')
        else:
            self._ckpt.restore(ckpt_path)
            self._logger.info('Checkpoint ID: {:d} restored from: {:s}'
                              .format(ckpt_id, str(self._ckpt_mngr.latest_checkpoint)))

        return

    def verify_checkpoint_restore(self):
        """
        Confirms that a checkpoint was loader properly.
        In graph mode, this can be called right after the self._ckpt.restore() call.
        In eager mode this should be called at least after one batch has been processed.
        :return: None
        """
        if self._ckpt_status is not None:
            self._logger('Verifying the checkpoint-restore operation.')
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



