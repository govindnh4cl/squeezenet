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

    # with tf.device(deploy_config.variables_device()):
    #     train_writer = tf.summary.FileWriter(args.model_dir, sess.graph)
    #     eval_dir = os.path.join(args.model_dir, 'eval')
    #     eval_writer = tf.summary.FileWriter(eval_dir, sess.graph)
    #     tf.summary.scalar('accuracy', train_metrics.accuracy)
    #     tf.summary.scalar('loss', model_dp.total_loss)
    #     all_summaries = tf.summary.merge_all()

    '''Model Checkpoints'''
    # saver = tf.train.Saver(max_to_keep=args.keep_last_n_checkpoints)
    # save_path = os.path.join(args.model_dir, 'model.ckpt')

    '''Model Initialization'''
    # last_checkpoint = tf.train.latest_checkpoint(args.model_dir)
    # if last_checkpoint:
    #     saver.restore(sess, last_checkpoint)
    # else:
    #     init_op = tf.group(tf.global_variables_initializer(),
    #                        tf.local_variables_initializer())
    #     sess.run(init_op)
    # starting_step = sess.run(global_step)

    '''Main Loop'''
    @tf.function  # For faster training speed
    def _train_step(nw, x_batch, opt):
        batch_x, batch_y = train_batch[0], train_batch[1]  # Get current batch samples

        with tf.GradientTape() as tape:
            batch_y_pred = nw(x_batch, training=True)  # Run prediction on batch
            loss_batch = loss_fn(batch_y, batch_y_pred)  # compute loss

        grads = tape.gradient(loss_batch, network.trainable_variables)  # compute gradient
        opt.apply_gradients(zip(grads, network.trainable_variables))  # Update weights

        return loss_batch

    batch_counter = 0
    # Loop over epochs
    for epoch_idx in range(cfg.max_train_epochs):
        start_time = time.time()
        batch_losses = list()

        # Loop over batches in the epoch
        for batch_idx, train_batch in enumerate(train_dataset):
            tf.summary.experimental.set_step(batch_counter)  # Set step for summaries

            batch_loss = _train_step(network, train_batch, optimizer)

            batch_losses.append(batch_loss)
            tf.summary.scalar('Train batch loss', batch_loss)

            print('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, np.mean(batch_losses)), end='')

            '''Checkpoint Hooks'''
            # if train_step % args.checkpoint_interval == 0:
            #     saver.save(sess, save_path, global_step)

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

            # tf.summary.text('Loss tryout', 'test_stirng')
            # tf.summary.text('Loss tryout', tf.as_string(tf.convert_to_tensor([batch_idx, 0.2])))
            batch_counter += 1

        print('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
            epoch_idx,
            np.mean(batch_losses),
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
        batch_losses = list()
        # Loop over batches in the epoch
        for batch_idx, train_batch in enumerate(train_dataset):
            tf.summary.experimental.set_step(batch_counter)  # Set step for summaries

            batch_x, batch_y = train_batch[0], train_batch[1]  # Get current batch samples
            batch_loss = model.train_on_batch(batch_x, batch_y)
            batch_losses.append(batch_loss)
            tf.summary.scalar('Train batch loss', batch_loss)

            print('\rEpoch {:3d} Training Loss {:f}'.format(epoch_idx, np.mean(batch_losses)), end='')
            batch_counter += 1

        print('\rEpoch {:3d} Training Loss {:f} Time {:.1f}s'.format(
            epoch_idx,
            np.mean(batch_losses),
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

    train_summary_writer = tf.summary.create_file_writer(cfg.tb_dir)

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


def run(args=None):
    args = arg_parsing.ArgParser().parse_args(args)

    cfg = EasyDict(vars(args))
    os.makedirs(args.log_dir, exist_ok=True)  # create directory if it doesn't exist
    cfg.tb_dir = os.path.join(cfg.log_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))  # Tensorboard directory

    _run(cfg)



