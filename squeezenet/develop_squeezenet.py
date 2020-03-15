import time
import numpy as np
import tensorflow as tf
import time

from my_logger import get_logger

from squeezenet import inputs
from squeezenet.config import get_config
from squeezenet.inputs import get_input_pipeline
from squeezenet.networks.squeezenet import Squeezenet_CIFAR, Squeezenet_Imagenet
from squeezenet import eval


class DevelopSqueezenet:
    def __init__(self, args):
        self.cfg = get_config(args)  # Get dictionary with configuration parameters
        self.logger = get_logger()
        self._pipeline = dict()

        # self.loss_fn = tf.keras.losses.CategoricalCrossentropy()  # Loss function
        # self.loss_fn = tf.losses.categorical_crossentropy  # Loss function
        self.loss_fn = lambda y, y_hat: -tf.math.reduce_sum(y * tf.math.log(y_hat + tf.keras.backend.epsilon()), axis=1)

        if self.cfg.misc.mode == 'train':
            self.net = self._set_network_for_training()
            self.opt = tf.keras.optimizers.Adam()  # Optimizer
        elif self.cfg.misc.mode == 'eval':
            self.logger.info("Loading model from directory: {:s}".format(self.cfg.directories.dir_model))
            if not tf.saved_model.contains_saved_model(self.cfg.directories.dir_model):
                raise OSError("Model directory: {:s} does not contain a saved model.")
            else:
                self.net = tf.saved_model.load(self.cfg.directories.dir_model)

        return

    @tf.function
    def _fwd_pass(self, batch_data):
        """
        Runs a forward pass on a batch
        :param batch_data: input samples in a batch
        :return: output predictions. Shape: (batch_size, 1)
        """
        batch_y_predicted = self.net(batch_data)  # Run prediction on batch

        return batch_y_predicted

    @tf.function  # For faster training speed
    def _train_step(self, batch_train):
        """

        :param batch_train:
        :return: loss_batch: A tensor scalar
        """
        batch_x, batch_y = batch_train[0], batch_train[1]  # Get current batch samples

        with tf.GradientTape() as tape:
            batch_y_pred = self.net(batch_x)  # Run prediction on batch
            loss_batch = tf.reduce_mean(self.loss_fn(batch_y, batch_y_pred))  # compute loss

        grads = tape.gradient(loss_batch, self.net.trainable_variables)  # compute gradient
        self.opt.apply_gradients(zip(grads, self.net.trainable_variables))  # Update weights

        return loss_batch

    def _train_tf(self, train_dataset, val_dataset):
        """

        :param train_dataset:
        :param val_dataset:
        :return:
        """
        self.logger.info('Training with Tensorflow API')

        # Model saver checkpoint
        if self.cfg.train.enable_save_best_model is True:
            best_model_value = tf.Variable(initial_value=np.inf, trainable=False, dtype=tf.float32)
            saver_ckpt = tf.train.Checkpoint(best_model_value=best_model_value)
            saver_ckpt_mngr = tf.train.CheckpointManager(checkpoint=saver_ckpt,
                                                         directory=self.cfg.directories.dir_ckpt_save_model,
                                                         max_to_keep=1)
            if saver_ckpt_mngr.latest_checkpoint:
                saver_ckpt_status = saver_ckpt.restore(saver_ckpt_mngr.latest_checkpoint)
                saver_ckpt_status.assert_consumed()

            self.logger.info('Saver checkpoint: Min validation loss: {:f}'.format(best_model_value.numpy()))

        # Model training checkpoints
        if self.cfg.train.enable_train_chekpoints:
            ckpt_counter = tf.Variable(initial_value=-1, trainable=False, dtype=tf.int64)  # Stores epoch ID
            ckpt = tf.train.Checkpoint(model=self.net,
                                       ckpt_counter=ckpt_counter,
                                       optimizer=self.opt)

            ckpt_mngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                                   directory=self.cfg.directories.dir_ckpt_train,
                                                   max_to_keep=1)

            # Checkpoint restoration
            if ckpt_mngr.latest_checkpoint:
                ckpt_status = ckpt.restore(ckpt_mngr.latest_checkpoint)
                self.logger.info('Training checkpoint: Restored from: {:s}'.format(ckpt_mngr.latest_checkpoint))
                self.logger.info('Training checkpoint: Checkpoint counter: {:d}'.format(ckpt_counter.numpy()))
            else:
                ckpt_status = None
                self.logger.info('Training checkpoint: Not found. Init weights from scratch.')

        self.net.training = True  # Enable training mode

        '''Main Loop'''
        last_sleep_time = time.time()
        batch_counter = tf.zeros(1, dtype=tf.int64)  # Overall batch-counter to serve as step for Tensorboard
        # Loop over epochs
        for epoch_idx in range(self.cfg.train.num_epochs):
            start_time = time.time()
            # Running average loss per sample during this epoch. Needed for printing loss during training
            running_loss = tf.keras.metrics.Mean()

            # Loop over batches in the epoch
            for batch_idx, train_batch in enumerate(train_dataset):
                tf.summary.experimental.set_step(batch_counter)  # Set step. Needed for summaries in Tensorboard

                batch_loss = self._train_step(train_batch)  # Tensor scalar loss for this batch

                running_loss.update_state(batch_loss)  # Update this batch's loss to
                tf.summary.scalar('Train loss', batch_loss)  # Log to tensorboard
                tf.summary.scalar('Train running-loss', running_loss.result())  # Log to tensorboard

                # Print status after each batch
                print('\rEpoch {:3d} Batch: {:d} Training Loss {:f}'.
                      format(epoch_idx, batch_idx, running_loss.result()), end='')

                batch_counter += 1  # Increment overall-batch-counter

                # Sleep intermittently to avoid burning down my machine
                if self.cfg.train.enable_intermittent_sleep and \
                        time.time() - last_sleep_time > self.cfg.train.sleep_interval:
                    self.logger.info('Sleeping for {:d} seconds.'.format(self.cfg.train.sleep_duration))
                    time.sleep(self.cfg.train.sleep_duration)
                    last_sleep_time = time.time()  # Reset

            self.logger.info('Epoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time))

            # Save checkpoint
            if self.cfg.train.enable_train_chekpoints:
                if ckpt_status is not None:
                    ckpt_status.assert_consumed()  # Verify the correctness of checkpoint loading
                    ckpt_status = None  # To prevent assert_consumed() from happening in next iteration

                ckpt.ckpt_counter.assign_add(1)  # Increment checkpoint id
                if int(ckpt.ckpt_counter) % self.cfg.train.checkpoint_interval == 0:
                    save_path = ckpt_mngr.save(checkpoint_number=int(ckpt.ckpt_counter))  # Save checkpoint
                    self.logger.info('Epoch {:3d} Saved checkpoint at {:s}'.format(epoch_idx, save_path))

            # TODO: time validation phase
            # TODO: Should we cover it with tf.no_gradient() of tf.stop_gradient() ?
            # Evaluate performance on validation set
            if self.cfg.validation.enable is True and epoch_idx % self.cfg.validation.validation_interval == 0:
                y_pred = np.nan * np.ones(shape=(self._pipeline['val'].count, self.cfg.dataset.num_classes), dtype=np.float32)
                y_true = np.nan * np.ones(shape=(self._pipeline['val'].count, self.cfg.dataset.num_classes), dtype=np.float32)

                # Loop over batches in the epoch
                idx = 0  # Index of samples processed so far
                for batch_idx, val_batch in enumerate(val_dataset):
                    batch_x, batch_y = val_batch[0], val_batch[1]  # Get current batch samples
                    batch_y_pred = self._fwd_pass(batch_x)

                    samples_in_batch = len(batch_y)
                    y_true[idx: idx + samples_in_batch] = batch_y
                    y_pred[idx: idx + samples_in_batch] = batch_y_pred
                    idx += samples_in_batch

                val_loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))
                val_acc = eval.get_categorical_accuracy(y_true, y_pred)
                tf.summary.scalar('Validation loss', val_loss)  # Log to tensorboard
                tf.summary.scalar('Validation accuracy', val_acc)  # Log to tensorboard
                self.logger.info('Epoch {:3d} Validation Loss: {:f} Categorical accuracy: {:.1f}%'
                                 .format(epoch_idx, val_loss, val_acc * 100))

            # Check if this is the best model so far, and if so, then save it
            if self.cfg.train.enable_save_best_model is True:
                old_value = best_model_value
                if self.cfg.train.best_model_criteria == 'train_loss':
                    new_value = running_loss
                elif self.cfg.train.best_model_criteria == 'val_loss':
                    new_value = val_loss
                else:
                    assert False  # Unsupported self.cfg.train.best_model_criteria

                if new_value < old_value:
                    self.logger.info('Saver checkpoint: Old: {:f} New: {:f}. Updating saved model in: {:s}'.
                                     format(old_value.read_value(), new_value, self.cfg.directories.dir_model))
                    best_model_value.assign(new_value)  # Update
                    saver_ckpt_mngr.save()  # Save checkpoint with newly found best value
                    tf.saved_model.save(self.net, self.cfg.directories.dir_model)  # Save model

        return

    def _train_keras(self, model, train_dataset):
        self.logger.info('Training with keras API')

        model.compile(loss='categorical_crossentropy', optimizer=self.opt)

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
                self.logger.info('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, running_loss.result()), end='')

                batch_counter += 1

            self.logger.info('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time))

        return

    def _set_network_for_training(self):
        """
        Network factory
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
        '''Inputs'''
        self._pipeline['train'] = get_input_pipeline(self.cfg, 'train')
        # self._pipeline['val'] = get_input_pipeline(self.cfg, 'val')
        self._pipeline['val'] = self._pipeline['train']  # TODO: disable this

        train_dataset = self._pipeline['train'].get_dataset()
        val_dataset = self._pipeline['val'].get_dataset()

        if self.cfg.train.enable_summary is True:
            train_summary_writer = tf.summary.create_file_writer(self.cfg.directories.dir_tb)
        else:
            train_summary_writer = tf.summary.create_noop_writer()

        with train_summary_writer.as_default():
            tf.summary.experimental.set_step(0)  # Set step for summaries
            if 1:
                self._train_tf(train_dataset, val_dataset)
            else:
                '''Model Creation'''
                model = net.get_keras_model()  # A keras model
                model.summary()
                _train_keras(model, train_dataset)

            self.logger.info('Training complete')

        return

    def _run_eval_mode(self):
        """
        Evaluates the model on dataset
        :return: None
        """
        if not tf.saved_model.contains_saved_model(self.cfg.directories.dir_model):
            raise FileNotFoundError('Could not find a saved model in directory: {:s}'
                                    .format(self.cfg.directories.dir_model))

        self.net = tf.saved_model.load(self.cfg.directories.dir_model)

        y_pred = np.nan * np.ones(shape=(self._pipeline['test'].count, self.cfg.dataset.num_classes), dtype=np.float32)
        y_true = np.nan * np.ones(shape=(self._pipeline['test'].count, self.cfg.dataset.num_classes), dtype=np.float32)

        self.logger.info('Running evaluation on dataset portion: {:s}'.format(self.cfg.eval.portion))
        self._pipeline[self.cfg.eval.portion] = get_input_pipeline(self.cfg, self.cfg.eval.portion)
        dataset = self._pipeline[self.cfg.eval.portion].get_dataset()

        # Loop over batches in the epoch
        idx = 0  # Index of samples processed so far
        for batch_idx, batch in enumerate(dataset):
            batch_x, batch_y = batch[0], batch[1]  # Get current batch samples
            batch_y_pred = self._fwd_pass(batch_x)

            samples_in_batch = len(batch_y)
            y_true[idx: idx + samples_in_batch] = batch_y
            y_pred[idx: idx + samples_in_batch] = batch_y_pred
            idx += samples_in_batch

        loss = tf.reduce_mean(self.loss_fn(y_true, y_pred))
        acc = eval.get_categorical_accuracy(y_true, y_pred)
        self.logger.info('Loss: {:f} Categorical accuracy: {:.1f}%'.format(loss, acc * 100))

        return

    def run(self):
        """
        Main entry point of DevelopSqueezenet class
        :return:
        """
        with tf.device(self.cfg.hardware.device):  # This does explicit device selection: cpu or gpu
            if self.cfg.misc.mode == 'train':
                self._run_train_mode()
            elif self.cfg.misc.mode == 'eval':
                self._run_eval_mode()


