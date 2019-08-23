import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

try:
    from squeezenet import arg_parsing
    from squeezenet import inputs
    from squeezenet import networks
    from squeezenet import metrics
except ImportError as e:
    print('Ignoring import error: {:s}'.format(str(e)))
    pass

logger = tf.get_logger()


def _run(args):
    # TODO: Make sure all tensors are created on the GPU

    network = networks.catalogue[args.network](args)

    # deploy_config = _configure_deployment(args.num_gpus)
    # sess = tf.Session(config=_configure_session())
    #
    # with tf.device(deploy_config.variables_device()):
    #     global_step = tf.train.create_global_step()
    #
    # with tf.device(deploy_config.optimizer_device()):
    #     optimizer = tf.train.AdamOptimizer(
    #         learning_rate=args.learning_rate
    #     )

    '''Inputs'''
    pipeline = inputs.Pipeline(args)  # Instantiate
    val_dataset = pipeline.get_val_dataset()
    train_dataset = pipeline.get_train_dataset()

    '''Model Creation'''
    model = network.build()  # A keras model

    ''' compile '''
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    '''Metrics'''
    # train_metrics = metrics.Metrics(
    #     labels=labels,
    #     clone_predictions=[clone.outputs['predictions']
    #                        for clone in model_dp.clones],
    #     device=deploy_config.variables_device(),
    #     name='training'
    # )
    # validation_metrics = metrics.Metrics(
    #     labels=labels,
    #     clone_predictions=[clone.outputs['predictions']
    #                        for clone in model_dp.clones],
    #     device=deploy_config.variables_device(),
    #     name='validation',
    #     padded_data=True
    # )

    '''Summaries'''
    # model.summary()

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
    assert isinstance(train_dataset, tf.data.Dataset)
    print_freq = 10  # Print after this many batches
    for batch_idx, train_batch in enumerate(train_dataset):
        x, y = train_batch[0], train_batch[1]
        loss = model.train_on_batch(x, y)

        if batch_idx % print_freq == 0:
            print('Step: {:5d}. Loss: {:}'.format(batch_idx, loss))
            # print('Batch labels: {:}'.format(np.argmax(y, axis=1).reshape((-1, ))))

        '''Summary Hook'''
        # if train_step % args.summary_interval == 0:
        #     results = sess.run(
        #         fetches={'accuracy': train_metrics.accuracy,
        #                  'summary': all_summaries},
        #         feed_dict=pipeline.training_data
        #     )
        #     # train_writer.add_summary(results['summary'], train_step)
        #     print('Train Step {:<5}:  {:>.4}'
        #           .format(train_step, results['accuracy']))

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

    print('Training complete')

def _configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.8)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)


def run(args=None):
    args = arg_parsing.ArgParser().parse_args(args)
    _run(args)



