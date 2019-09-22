import time
import tensorflow as tf

from my_logger import get_logger
from squeezenet.arg_parsing import parse_args
from squeezenet import inputs
from squeezenet import networks
from squeezenet.config import get_config
from squeezenet.networks.squeezenet import Squeezenet_CIFAR, Squeezenet_Imagenet


def _train_tf(cfg, net, train_dataset):
    logger = get_logger()
    logger.info('Training with Tensorflow API')
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # Model checkpoints
    ckpt_counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(ckpt_counter=ckpt_counter,
                               optimizer=optimizer, model=net)

    ckpt_mngr = tf.train.CheckpointManager(checkpoint=ckpt,
                                           directory=cfg.directories.dir_ckpt,
                                           max_to_keep=3)

    # Checkpoint restoration
    if ckpt_mngr.latest_checkpoint:
        ckpt_status = ckpt.restore(ckpt_mngr.latest_checkpoint)
        logger.info('Restored checkpoint from: {:s}'.format(ckpt_mngr.latest_checkpoint))
    else:
        ckpt_status = None
        logger.info('No checkpoint found. Starting from scratch.')

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
    for epoch_idx in range(cfg.train.num_epochs):
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

            batch_counter += 1  # Increment overall-batch-counter

        # Validate the checkpoint loading
        if ckpt_status is not None:
            ckpt_status.assert_consumed()  # Sanity check that checkpoint loading was error-free

        # Save checkpoint
        if int(ckpt.ckpt_counter) % 5 == 0:
            ckpt.ckpt_counter.assign_add(1)  # Increment checkpoint id
            save_path = ckpt_mngr.save(checkpoint_number=int(ckpt.ckpt_counter))  # Save checkpoint
            logger.info('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s. Saved checkpoint at {:s}'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time,
                save_path))
        else:
            logger.info('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
                epoch_idx,
                running_loss.result(),
                time.time() - start_time))

    return


def _train_keras(cfg, model, train_dataset):
    logger = get_logger()
    logger.info('Training with keras API')
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    '''Main Loop'''
    batch_counter = 0
    # Loop over epochs
    for epoch_idx in range(cfg.train.num_epochs):
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

        logger.info('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
            epoch_idx,
            running_loss.result(),
            time.time() - start_time))

    return


def _get_network(cfg):
    """
    Network factory
    :param cfg:
    :return:
    """
    if cfg.dataset.dataset == 'imagenet':
        net = Squeezenet_Imagenet(cfg)
    elif cfg.dataset.dataset == 'cifar10':
        net = Squeezenet_CIFAR(cfg)

    return net


def _run(cfg):
    logger = get_logger()
    net = _get_network(cfg)

    '''Inputs'''
    pipeline = inputs.Pipeline(cfg)  # Instantiate
    val_dataset = pipeline.get_val_dataset()
    train_dataset = pipeline.get_train_dataset()
    assert isinstance(train_dataset, tf.data.Dataset)

    train_summary_writer = tf.summary.create_file_writer(cfg.directories.dir_tb)

    with train_summary_writer.as_default():
        tf.summary.experimental.set_step(0)  # Set step for summaries
        if 1:
            _train_tf(cfg, net, train_dataset)
        else:
            '''Model Creation'''
            model = net.get_keras_model()  # A keras model
            model.summary()
            _train_keras(cfg, model, train_dataset)

        logger.info('Training complete')


def run():
    args = parse_args()
    cfg = get_config(args)  # Get dictionary with configuration parameters

    with tf.device(cfg.hardware.device):  # This does explicit device selection: cpu or gpu
        _run(cfg)



