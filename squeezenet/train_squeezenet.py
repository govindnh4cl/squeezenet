import os
import time
from easydict import EasyDict
import numpy as np
import tensorflow as tf

try:
    from squeezenet import arg_parsing
    from squeezenet import inputs
    from squeezenet import networks
    from squeezenet import metrics
except ImportError as e:
    print('Ignoring import error: {:s}'.format(str(e)))
    pass

logger = tf.get_logger()


def _train_tf(cfg, network, train_dataset):
    print('Training with Tensorflow API')
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Model checkpoints
    ckpt_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(ckpt_counter=ckpt_counter,
                               optimizer=optimizer, model=network)

    ckpt_mngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                           directory=cfg.dir_ckpt,
                                           max_to_keep=3)

    # Checkpoint restoration
    ckpt.restore(ckpt_mngr.latest_checkpoint)
    if ckpt_mngr.latest_checkpoint:
        print('Restoring checkpoint from: {:s}'.format(ckpt_mngr.latest_checkpoint))
        ckpt.restore(ckpt_mngr.latest_checkpoint)
    else:
        print('No checkpoint found. Initializing from scratch.')

    @tf.function  # For faster training speed
    def _train_step(nw, batch_train, opt):
        batch_x, batch_y = batch_train[0], batch_train[1]  # Get current batch samples

        with tf.GradientTape() as tape:
            batch_y_pred = nw(batch_x, training=True)  # Run prediction on batch
            loss_batch = loss_fn(batch_y, batch_y_pred)  # compute loss

        grads = tape.gradient(loss_batch, network.trainable_variables)  # compute gradient
        opt.apply_gradients(zip(grads, network.trainable_variables))  # Update weights

        return loss_batch

    '''Main Loop'''
    batch_counter = tf.zeros(1, dtype=tf.int64)
    # Loop over epochs
    for epoch_idx in range(cfg.max_train_epochs):
        start_time = time.time()
        running_loss = tf.keras.metrics.Mean()  # Running loss per sample

        # Loop over batches in the epoch
        for batch_idx, train_batch in enumerate(train_dataset):
            tf.summary.experimental.set_step(batch_counter)  # Set step. Needed for summaries

            batch_loss = _train_step(network, train_batch, optimizer)  # batch_loss is a Tensor scalar

            running_loss.update_state(batch_loss)
            tf.summary.scalar('Train loss', batch_loss)
            tf.summary.scalar('Train running-loss', running_loss.result())
            # print('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, running_loss.result()), end='')

            # sess.run(train_metrics.reset_op)
            #
            # '''Eval Hook'''
            # if train_step % args.validation_interval == 0:
            #     while True:
            #         try:
            #             sess.run(
            #                 fetches=validation_metrics.update_op,
            #                 feed_dict=pipeline.validation_data
            #             )
            #         except tf.errors.OutOfRangeError:
            #             break
            #     results = sess.run({'accuracy': validation_metrics.accuracy})
            #
            #     print('Evaluation Step {:<5}:  {:>.4}'
            #           .format(train_step, results['accuracy']))
            #
            #     # summary = tf.Summary(value=[
            #     #     tf.Summary.Value(tag='accuracy', simple_value=results['accuracy']),
            #     # ])
            #     # eval_writer.add_summary(summary, train_step)
            #     sess.run(validation_init_op)  # Reinitialize dataset and metrics

            batch_counter += 1

        # Save checkpoint
        if int(ckpt.ckpt_counter) % 1 == 0:
            ckpt.ckpt_counter.assign_add(1)  # Increment checkpoint id
            save_path = ckpt_mngr.save(checkpoint_number=int(ckpt.ckpt_counter))  # Save checkpoint
            print('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s. Saved checkpoint at {:s}'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time,
                save_path))
        else:
            print('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time))

    return


def _train_keras(cfg, model, train_dataset):
    print('Training with keras API')
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    '''Main Loop'''
    batch_counter = 0
    # Loop over epochs
    for epoch_idx in range(cfg.max_train_epochs):
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
            print('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, running_loss.result()), end='')

            batch_counter += 1

        print('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
            epoch_idx,
            running_loss.result(),
            time.time() - start_time))

    return


def _run(cfg):
    # TODO: Make sure all tensors are created on the GPU

    network = networks.catalogue[cfg.network](cfg)

    '''Inputs'''
    pipeline = inputs.Pipeline(cfg)  # Instantiate
    val_dataset = pipeline.get_val_dataset()
    train_dataset = pipeline.get_train_dataset()
    assert isinstance(train_dataset, tf.data.Dataset)

    train_summary_writer = tf.summary.create_file_writer(cfg.dir_tb)

    with train_summary_writer.as_default():
        tf.summary.experimental.set_step(0)  # Set step for summaries
        with tf.device('/GPU:0'):  # Make sure that we're using GPU
            if 1:
                _train_tf(cfg, network, train_dataset)
            else:
                '''Model Creation'''
                model = network.get_keras_model()  # A keras model
                model.summary()
                _train_keras(cfg, model, train_dataset)

        print('Training complete')


def _configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.8)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)

def _get_config(args):
    cfg = EasyDict(vars(args))
    cfg.dir_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfg.dir_tb = os.path.join(cfg.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))  # Tensorboard directory
    cfg.dir_ckpt = os.path.join(cfg.dir_repo, 'checkpoints')
    return cfg

def run(args=None):
    args = arg_parsing.ArgParser().parse_args(args)
    cfg = _get_config(args)

    os.makedirs(cfg.log_dir, exist_ok=True)  # create directory if it doesn't exist
    os.makedirs(cfg.dir_ckpt, exist_ok=True)  # create directory if it doesn't exist
    _run(cfg)



