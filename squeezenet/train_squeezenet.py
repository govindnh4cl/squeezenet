import time
import numpy as np
import tensorflow as tf

from my_logger import get_logger

from squeezenet import inputs
from squeezenet.config import get_config
from squeezenet.networks.squeezenet import Squeezenet_CIFAR, Squeezenet_Imagenet
from squeezenet import eval


class DevelopSqueezenet:
    def __init__(self, args):

        self.cfg = get_config(args)  # Get dictionary with configuration parameters
        pass

    def _train_tf(self, net, pipeline, train_dataset, val_dataset):
        """

        :param net:
        :param train_dataset:
        :param val_dataset:
        :return:
        """
        logger = get_logger()
        logger.info('Training with Tensorflow API')
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        # Model checkpoints
        ckpt_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(ckpt_counter=ckpt_counter,
                                   optimizer=optimizer, model=net)

        ckpt_mngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                               directory=self.cfg.directories.dir_ckpt,
                                               max_to_keep=3)

        # Checkpoint restoration
        if ckpt_mngr.latest_checkpoint:
            ckpt_status = ckpt.restore(ckpt_mngr.latest_checkpoint)
            logger.info('Restored checkpoint from: {:s}'.format(ckpt_mngr.latest_checkpoint))
        else:
            ckpt_status = None
            logger.info('No checkpoint found. Starting from scratch.')

        @tf.function
        def _fwd_pass(nw, batch_data):
            """
            Runs a forward pass on a batch
            :param nw: Network object
            :param batch_x: input samples in a batch
            :return: output predictions. Shape: (batch_size, 1)
            """
            batch_y_predicted = nw(batch_data, training=False)  # Run prediction on batch

            return batch_y_predicted

        @tf.function  # For faster training speed
        def _train_step(nw, batch_train, opt):
            batch_x, batch_y = batch_train[0], batch_train[1]  # Get current batch samples

            with tf.GradientTape() as tape:
                batch_y_pred = nw(batch_x, training=True)  # Run prediction on batch
                loss_batch = loss_fn(batch_y, batch_y_pred)  # compute loss

            grads = tape.gradient(loss_batch, net.trainable_variables)  # compute gradient
            opt.apply_gradients(zip(grads, net.trainable_variables))  # Update weights

            return loss_batch

        '''Main Loop'''
        batch_counter = tf.zeros(1, dtype=tf.int64)  # Overall batch-counter to serve as step for Tensorboard
        # Loop over epochs
        for epoch_idx in range(self.cfg.train.num_epochs):
            start_time = time.time()
            # Running average loss per sample during this epoch. Needed for printing loss during training
            running_loss = tf.keras.metrics.Mean()

            # Loop over batches in the epoch
            for batch_idx, train_batch in enumerate(train_dataset):
                tf.summary.experimental.set_step(batch_counter)  # Set step. Needed for summaries in Tensorboard

                batch_loss = _train_step(net, train_batch, optimizer)  # Tensor scalar loss for this batch

                running_loss.update_state(batch_loss)  # Update this batch's loss to
                tf.summary.scalar('Train loss', batch_loss)  # Log to tensorboard
                tf.summary.scalar('Train running-loss', running_loss.result())  # Log to tensorboard
                # print('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, running_loss.result()), end='')

                batch_counter += 1  # Increment overall-batch-counter

            # Verify the correctness of checkpoint loading
            if ckpt_status is not None:
                ckpt_status.assert_consumed()  # Sanity check that checkpoint loading was error-free

            # Save checkpoint
            if int(ckpt.ckpt_counter) % self.cfg.train.checkpoint_interval == 0:
                save_path = ckpt_mngr.save(checkpoint_number=int(ckpt.ckpt_counter))  # Save checkpoint
                logger.info('Epoch {:3d} Training Loss {:f} Time {:.1f}s. Saved checkpoint at {:s}'.format(
                    epoch_idx,
                    running_loss.result(),
                    time.time() - start_time,
                    save_path))
            else:
                logger.info('Epoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
                    epoch_idx,
                    running_loss.result(),
                    time.time() - start_time))
            ckpt.ckpt_counter.assign_add(1)  # Increment checkpoint id

            # Evaluate performance on validation set
            if self.cfg.validation.enable is True and epoch_idx % self.cfg.validation.validation_interval == 0:
                y_pred = np.nan * np.ones(shape=(pipeline.count_val, self.cfg.dataset.num_classes), dtype=np.float32)
                y_true = np.nan * np.ones(shape=(pipeline.count_val, self.cfg.dataset.num_classes), dtype=np.float32)

                # Loop over batches in the epoch
                idx = 0  # Index of samples processed so far
                for batch_idx, val_batch in enumerate(val_dataset):
                    batch_x, batch_y = val_batch[0], val_batch[1]  # Get current batch samples
                    batch_y_pred = _fwd_pass(net, batch_x)

                    samples_in_batch = len(batch_y)
                    y_true[idx: idx + samples_in_batch] = batch_y
                    y_pred[idx: idx + samples_in_batch] = batch_y_pred
                    idx += samples_in_batch

                val_loss = loss_fn(y_true, y_pred)
                val_acc = eval.get_categorical_accuracy(y_true, y_pred)
                logger.info('Epoch {:3d} Validation Loss: {:f} Categorical accuracy: {:.1f}%'
                            .format(epoch_idx, val_loss, val_acc * 100))
        return

    def _train_keras(self, model, train_dataset):
        logger = get_logger()
        logger.info('Training with keras API')
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        model.compile(loss='categorical_crossentropy', optimizer='adam')

        '''Main Loop'''
        batch_counter = 0
        # Loop over epochs
        for epoch_idx in range(self.cfg.train.num_epochs):
            start_time = time.time()
            running_loss = tf.keras.metrics.Mean()  # Running loss per sample
            # Loop over batches in the epoch
            for batch_idx, train_batch in enumerate(train_dataset):
                tf.summary.experimental.set_step(batch_counter)  # Set step for summaries

                batch_x, batch_y = train_batch[0], train_batch[1]  # Get current batch samples
                batch_loss = model.train_on_batch(batch_x, batch_y)

                running_loss.update_state(batch_loss)
                tf.summary.scalar('Train loss', batch_loss)
                tf.summary.scalar('Train running-loss', running_loss.result())
                logger.info('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, running_loss.result()), end='')

                batch_counter += 1

            logger.info('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time))

        return

    def _get_network(self):
        """
        Network factory
        :param cfg:
        :return:
        """
        if self.cfg.dataset.dataset == 'imagenet':
            net = Squeezenet_Imagenet(self.cfg)
        elif self.cfg.dataset.dataset == 'cifar10':
            net = Squeezenet_CIFAR(self.cfg)

        return net

    def _run_train_mode(self):
        """
        Peforms training of the squeezement
        :return: None
        """
        logger = get_logger()
        net = self._get_network()

        '''Inputs'''
        pipeline = inputs.Pipeline(self.cfg)  # Instantiate
        train_dataset = pipeline.get_train_dataset()
        val_dataset = pipeline.get_val_dataset()

        train_summary_writer = tf.summary.create_file_writer(self.cfg.directories.dir_tb)

        with train_summary_writer.as_default():
            tf.summary.experimental.set_step(0)  # Set step for summaries
            if 1:
                self._train_tf(net, pipeline, train_dataset, val_dataset)
            else:
                '''Model Creation'''
                model = net.get_keras_model()  # A keras model
                model.summary()
                _train_keras(model, train_dataset)

            logger.info('Training complete')

        return

    def _run_eval_mode(self):
        """
        Evaluates the model on dataset
        :return: None
        """
        # TODO: implementation
        raise NotImplementedError

    def run(self):
        """
        Main entry point of DevelopSqueezenet class
        :return:
        """
        if self.cfg.misc.mode == 'train':
            with tf.device(self.cfg.hardware.device):  # This does explicit device selection: cpu or gpu
                self._run_train_mode()

        if self.cfg.misc.mode == 'eval':
            with tf.device(self.cfg.hardware.device):  # This does explicit device selection: cpu or gpu
                self._run_eval_mode()


