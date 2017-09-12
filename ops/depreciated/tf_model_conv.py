"""DEPRECIATED."""

import os
import re
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from ops.data_loader import inputs
from ops.tf_fun import softmax_cost, fscore, make_dir


def train_and_eval(config):
    """Train and evaluate the model."""
    print 'Model directory: %s' % config.model_output
    print 'Running model: %s' % config.model_type
    if config.model_type == 'fully_connected':
        from models.fully_connected import model_struct
    elif config.model_type == 'vgg_feature_model':
        from models.vgg_feature_model import model_struct
    else:
        raise Exception

    # Prepare model training
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.model_type + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.model_output, dt_dataset)  # timestamp this run
    config.summary_dir = os.path.join(
        config.train_summaries, config.model_output, dt_dataset)
    dir_list = [config.train_checkpoint, config.summary_dir]
    [make_dir(d) for d in dir_list]

    # Prepare model inputs
    train_data = os.path.join(config.tfrecord_dir, 'train.tfrecords')
    validation_data = os.path.join(config.tfrecord_dir, 'val.tfrecords')
    feat_mean = 0  # np.mean(np.load(config.mean_file)['feat_list'])

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(
            tfrecord_file=train_data,
            batch_size=config.train_batch,
            num_feats=config.n_features,
            im_size=config.resize,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            feat_mean_value=feat_mean)
        val_images, val_labels = inputs(
            tfrecord_file=validation_data,
            batch_size=config.train_batch,
            num_feats=config.n_features,
            im_size=config.resize,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            num_epochs=config.epochs,
            feat_mean_value=feat_mean)
        tf.summary.image('train images', train_images)
        tf.summary.image('validation images', val_images)

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:

            model = model_struct()
            train_mode = tf.get_variable(name='training', initializer=True)
            model.build(
                features=train_images,
                output_categories=len(config.labels.keys()),
                train_mode=train_mode, batchnorm=config.batch_norm)

            # Prepare the cost function
            cost = softmax_cost(
                model.res_logits, train_labels, ratio=config.ratio)
            train_op = tf.train.AdamOptimizer(config.lr).minimize(cost)

            tf.summary.scalar("cost", cost)

            train_score = fscore(
                model.prob, train_labels)  # training accuracy
            tf.summary.scalar("training f-score", train_score)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                # Validation graph is the same as training except no batchnorm
                val_model = model_struct()
                val_model.build(
                    features=val_images,
                    output_categories=len(config.labels.keys()))

                # Calculate validation accuracy
                val_score = fscore(val_model.prob, val_labels)
                tf.summary.scalar("validation f-score", val_score)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(config.train_checkpoint, config)
    step, val_max, losses = 0, 0, []

    try:
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_acc = sess.run(
                [train_op, cost, train_score])
            losses.append(loss_value)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 100 == 0 and step % 10 == 0:
                if validation_data is not False:
                    _, val_acc = sess.run([train_op, val_score])
                else:
                    val_acc -= 1  # Store every checkpoint

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.train_batch / duration, float(duration),
                    train_acc, val_acc, config.summary_dir))

                # Save the model checkpoint if it's the best yet
                if val_acc > val_max:
                    saver.save(
                        sess, os.path.join(
                            config.train_checkpoint,
                            'model_' + str(step) + '.ckpt'), global_step=step)
                    # Store the new max validation accuracy
                    val_max = val_acc

            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training accuracy = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), train_acc))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()
        np.save(os.path.join(config.tfrecord_dir, 'training_loss'), losses)
    coord.join(threads)
    sess.close()
