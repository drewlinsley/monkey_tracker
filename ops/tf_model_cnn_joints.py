import os
import re
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from ops.data_loader_joints import inputs
from ops.tf_fun import regression_mse, correlation, make_dir, \
    fine_tune_prepare_layers, ft_optimizer_list
from ops.utils import get_dt


def train_and_eval(config, use_train=True):
    """Train and evaluate the model."""
    print 'Model directory: %s' % config.model_output
    print 'Running model: %s' % config.model_type
    if config.model_type == 'fully_connected_mlp':
        from models.fully_connected_mlp import model_struct
    elif config.model_type == 'fully_connected_mlp_2l':
        from models.fully_connected_mlp_2l import model_struct
    elif config.model_type == 'fully_connected_conv':
        from models.fully_connected_conv import model_struct
    elif config.model_type == 'vgg_feature_model':
        from models.vgg_feature_model import model_struct
    elif config.model_type == 'vgg_regression_model':
        from models.vgg_regression_model import model_struct
    elif config.model_type == 'vgg_regression_model_4fc':
        from models.vgg_regression_model_4fc import model_struct
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

    # Prepare model inputs
    if use_train:
        print 'Testing on training dataset %s' % os.path.join(
            config.tfrecord_dir, config.train_tfrecords)
        validation_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    else:
        validation_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        val_images, val_labels = inputs(
            tfrecord_file=validation_data,
            batch_size=1,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=1,
            image_target_size=config.image_target_size,
            image_input_size=config.image_input_size,
            maya_conversion=config.maya_conversion,
            occlusions=self.occlusion_dir
            )

    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:

            model = model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.initialize_layers)
            train_mode = tf.get_variable(name='training', initializer=True)
            model.build(
                rgb=val_images,
                output_shape=config.num_classes,
                train_mode=train_mode,
                batchnorm=config.batch_norm)

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
    if config.resume_from_checkpoint is not None:
        saver.restore(config.resume_from_checkpoint, sess)

    try:
        while not coord.should_stop():
            start_time = time.time()
            yhat, im, ytrue = sess.run([model.fc8, val_images, val_labels])
            np.save('/media/data_cifs/monkey_tracking/batches/test/im', im)
            np.save('/media/data_cifs/monkey_tracking/batches/test/yhat', yhat)
            np.save('/media/data_cifs/monkey_tracking/batches/test/ytrue', ytrue)

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()

        dt_stamp = get_dt()  # date-time stamp
        np.save(
            os.path.join(
                config.tfrecord_dir, '%straining_loss' % dt_stamp), losses)
    coord.join(threads)
    sess.close()
