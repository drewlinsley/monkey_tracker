import os
import re
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from ops.data_loader_joints import inputs
from ops.tf_fun import regression_mse, correlation, make_dir, \
    fine_tune_prepare_layers, ft_optimizer_list, softmax_cost
from ops.utils import get_dt


def train_and_eval(config):
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
    elif config.model_type == 'cnn_multiscale':
        from models.cnn_multiscale import model_struct
    elif config.model_type == 'cnn_multiscale_low_high_res':
        from models.cnn_multiscale_low_high_res import model_struct
    elif config.model_type == 'cnn_multiscale_low_high_res_mid_loss':
        from models.cnn_multiscale_low_high_res_mid_loss import model_struct
    elif config.model_type == 'test':
        from models.test import model_struct
    else:
        raise RuntimeError('Cannot understand what kind of model you want to run.')

    # Prepare model training
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.model_type + '_' + dt_stamp + '/'
    config.train_checkpoint = os.path.join(
        config.model_output, dt_dataset)  # timestamp this run
    config.summary_dir = os.path.join(
        config.train_summaries, dt_dataset)
    results_dir = os.path.join(config.npy_dir, dt_stamp)
    print 'Saving Dmurphy\'s online updates to: %s' % results_dir
    dir_list = [config.train_checkpoint, config.summary_dir, results_dir]
    [make_dir(d) for d in dir_list]

    # Prepare model inputs
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    validation_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels, train_occlusions = inputs(
            tfrecord_file=train_data,
            batch_size=config.train_batch,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=config.epochs,
            image_target_size=config.image_target_size,
            image_input_size=config.image_input_size,
            maya_conversion=config.maya_conversion,
            max_value=config.max_depth,
            return_occlusions=config.occlusion_dir,
            normalize_labels=config.normalize_labels
            )
        val_images, val_labels, val_occlusions = inputs(
            tfrecord_file=validation_data,
            batch_size=config.validation_batch,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=config.epochs,
            image_target_size=config.image_target_size,
            image_input_size=config.image_input_size,
            maya_conversion=config.maya_conversion,
            max_value=config.max_depth,
            return_occlusions=config.occlusion_dir,
            normalize_labels=config.normalize_labels
            )
        tf.summary.image(
            'train images', tf.cast(train_images, tf.float32))
        tf.summary.image(
            'validation images', tf.cast(val_images, tf.float32))

    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:

            model = model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.initialize_layers)
            train_mode = tf.get_variable(name='training', initializer=True)
            model.build(
                rgb=train_images,
                output_shape=config.num_classes,
                train_mode=train_mode,
                batchnorm=config.batch_norm)

            # Prepare the loss functions:::
            loss_list, loss_label = [], []
            # 1. High-res head
            if config.model_type == 'cnn_multiscale_low_high_res_mid_loss':
                loss_list += [tf.nn.l2_loss(
                    model.high_feature_encoder_joints - train_labels)]
                loss_label += ['high-res head']
                # 2. Low-res head
                loss_list += [tf.nn.l2_loss(
                    model.low_feature_encoder_joints - train_labels)]
                loss_label += ['low-res head']
            # 3. Combined head loss -- joints
            loss_list += [tf.nn.l2_loss(
                model.fc8 - train_labels)]
            loss_label += ['combined head']
            # 4. Combined head loss -- occlusions
            # loss_list += [tf.nn.l2_loss(
            #     model.fc8_occlusion - train_occlusions)]
            loss_list += [tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=train_occlusions,
                    logits=model.fc8_occlusion))]
            loss_label += ['occlusion head']
            loss = tf.add_n(loss_list)

            # Add wd if necessary
            if config.wd_penalty is not None:
                _, l2_wd_layers = fine_tune_prepare_layers(
                    tf.trainable_variables(), config.wd_layers)
                l2_wd_layers = [
                    x for x in l2_wd_layers if 'biases' not in x.name]
                # import ipdb;ipdb.set_trace()
                loss += (
                    config.wd_penalty * tf.add_n(
                        [tf.nn.l2_loss(x) for x in l2_wd_layers]))

            # other_opt_vars, ft_opt_vars = fine_tune_prepare_layers(
            #     tf.trainable_variables(), config.fine_tune_layers)

            # train_op, _ = ft_optimizer_list(
            #     loss, [other_opt_vars, ft_opt_vars],
            #     optimizer,
            #     [config.hold_lr, config.lr])

            if config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer
            elif config.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer
            elif config.optimizer == 'rms':
                optimizer = tf.train.RMSPropOptimizer
            else:
                raise 'Unidentified optimizer'

            # Gradient Descent
            optimizer = optimizer(
                config.lr)
            # Op to calculate every variable gradient
            # grads = optimizer.compute_gradients(
            #     loss, tf.trainable_variables())
            # grads = [(tf.clip_by_norm(
            #     g, 8), v) for g, v in grads if g is not None]
            # Op to update all variables according to their gradient
            # train_op = optimizer.apply_gradients(
            #     grads_and_vars=grads)

            # Summarize all gradients and weights
            # [tf.summary.histogram(
            #     var.name + '/gradient', grad)
            #     for grad, var in grads if grad is not None]
            train_op = optimizer.minimize(loss)

            # Summarize scores
            train_score, _ = correlation(
                model.fc8, train_labels)  # training accuracy
            tf.summary.scalar("training correlation", train_score)
            [tf.summary.scalar(lab, il) for lab, il in zip(
                loss_label, loss_list)]
            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                # Validation graph is the same as training except no batchnorm
                val_model = model_struct()
                val_model.build(
                    rgb=val_images,
                    output_shape=config.num_classes),

                # Calculate validation accuracy
                val_score = tf.nn.l2_loss(val_model.fc8 - val_labels)
                tf.summary.scalar("validation mse", val_score)

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
    step, losses = 0, []
    train_acc = 0
    if config.resume_from_checkpoint is not None:
        print 'Resuming training from checkpoint: %s' % config.resume_from_checkpoint
        saver.restore(sess, config.resume_from_checkpoint)
    try:
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_acc, im, yhat, ytrue, occhat, occtrue = sess.run([
                train_op,
                loss,
                train_score,
                train_images,
                model.fc8,
                train_labels,
                model.fc8_occlusion,
                train_occlusions
            ])
            # import scipy.misc
            # np.save('/media/data_cifs/monkey_tracking/batches/test/im', im)
            # np.save('/media/data_cifs/monkey_tracking/batches/test/yhat', yhat)
            # np.save('/media/data_cifs/monkey_tracking/batches/test/ytrue', ytrue)

            losses.append(loss_value)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % config.steps_before_validation == 0:
                if validation_data is not False:
                    val_acc, val_pred, val_ims = sess.run(
                        [val_score, val_model.fc8, val_images])

                    np.savez(
                        os.path.join(
                            config.model_output, '%s_val_coors' % step),
                        val_pred=val_pred, val_ims=val_ims)
                else:
                    val_acc = -1  # Store every checkpoint

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy attach 9177
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training r = %s | '
                    'Validation r = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.train_batch / duration, float(duration),
                    train_acc, val_acc, config.summary_dir))

                # Save the model checkpoint if it's the best yet
                if config.normalize_labels:
                    normalize_vec = np.asarray(
                        config.image_target_size[:2] + [config.max_depth]).reshape(
                        1, -1).repeat(23, axis=0).reshape(1, -1)
                    yhat *= normalize_vec
                    ytrue *= normalize_vec
                np.save(
                    os.path.join(results_dir, 'im_%s' % step), im)
                np.save(
                    os.path.join(results_dir, 'yhat_%s' % step), yhat)
                np.save(
                    os.path.join(results_dir, 'ytrue_%s' % step), ytrue)
                np.save(
                    os.path.join(results_dir, 'occhat_%s' % step), occhat)
                np.save(
                    os.path.join(results_dir, 'occtrue_%s' % step), occtrue)
                saver.save(
                    sess, os.path.join(
                        config.train_checkpoint,
                        'model_' + str(step) + '.ckpt'), global_step=step)

            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training F = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), train_acc))
            # End iteration
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
