import os
import re
from datetime import datetime
import numpy as np
import tensorflow as tf
from ops.data_loader_joints import inputs
from ops import tf_fun
from ops.utils import get_dt, import_cnn
import argparse
from config import monkeyConfig
from visualization import monkey_mosaic


def train_and_eval(
        train_data,
        validation_data,
        config,
        uniform_batch_size=6,
        swap_datasets=False):
    """Train and evaluate the model."""

    # Import your model
    print 'Model directory: %s' % config.model_output
    print 'Running model: %s' % config.model_type
    model_file = import_cnn(config.model_type)

    # Prepare model training
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = '%s_%s' % (config.model_type, dt_stamp)
    if config.selected_joints is not None:
        dt_dataset = '_%s' % (config.selected_joints) + dt_dataset
    config.train_checkpoint = os.path.join(
        config.model_output, dt_dataset)  # timestamp this run
    config.summary_dir = os.path.join(
        config.train_summaries, dt_dataset)
    results_dir = os.path.join(config.npy_dir, dt_dataset)
    print 'Saving Dmurphy\'s online updates to: %s' % results_dir
    dir_list = [config.train_checkpoint, config.summary_dir, results_dir]
    [tf_fun.make_dir(d) for d in dir_list]

    # Prepare model inputs
    if train_data is None:
        train_data = os.path.join(
            config.tfrecord_dir,
            config.train_tfrecords)

    if validation_data is None:
        validation_data = os.path.join(
            config.tfrecord_dir,
            config.val_tfrecords)
    if swap_datasets:
        t_train = np.copy(train_data)
        train_data = validation_data
        validation_data = str(t_train)

    # Prepare data on CPU
    config.train_batch = uniform_batch_size
    with tf.device('/cpu:0'):
        train_data_dict = inputs(
            tfrecord_file=train_data,
            batch_size=config.train_batch,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=None,
            image_target_size=config.image_target_size,
            image_input_size=config.image_input_size,
            maya_conversion=config.maya_conversion,
            max_value=config.max_depth,
            normalize_labels=config.normalize_labels,
            aux_losses=config.aux_losses,
            selected_joints=config.selected_joints,
            joint_names=config.joint_order,
            num_dims=config.num_dims,
            keep_dims=config.keep_dims,
            mask_occluded_joints=config.mask_occluded_joints,
            background_multiplier=config.background_multiplier)

        val_data_dict = inputs(
            tfrecord_file=validation_data,
            batch_size=config.train_batch,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=None,
            image_target_size=config.image_target_size,
            image_input_size=config.image_input_size,
            maya_conversion=config.maya_conversion,
            max_value=config.max_depth,
            normalize_labels=config.normalize_labels,
            aux_losses=config.aux_losses,
            selected_joints=config.selected_joints,
            joint_names=config.joint_order,
            num_dims=config.num_dims,
            keep_dims=config.keep_dims,
            mask_occluded_joints=config.mask_occluded_joints,
            background_multiplier=config.background_multiplier)

        # Check output_shape
        if config.selected_joints is not None:
            print 'Targeting joint: %s' % config.selected_joints
            joint_shape = len(config.selected_joints) * config.keep_dims
            if (config.num_classes // config.keep_dims) > (joint_shape):
                print 'New target size: %s' % joint_shape
                config.num_classes = joint_shape

    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            print 'Creating training graph:'
            model = model_file.model_struct(
                vgg16_npy_path=config.vgg16_weight_path)
            train_mode = tf.get_variable(name='training', initializer=True)
            model.build(
                rgb=train_data_dict['image'],
                target_variables=train_data_dict,
                train_mode=train_mode,
                batchnorm=config.batch_norm)

            # Prepare the loss functions:::
            loss_list, loss_label = [], []
            if 'label' in train_data_dict.keys():
                # 1. Joint localization loss
                if config.calculate_per_joint_loss:
                    label_loss, use_joints, joint_variance = tf_fun.thomas_l1_loss(
                        model=model,
                        train_data_dict=train_data_dict,
                        config=config)
                    loss_list += [label_loss]
                else:
                    loss_list += [tf.add_n([tf.nn.l2_loss(
                        model[x] - train_data_dict['label']) for x in model.joint_label_output_keys])]
                loss_label += ['combined head']
            if 'occlusion' in train_data_dict.keys():
                # 2. Auxillary losses
                # a. Occlusion
                loss_list += [tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=train_data_dict['occlusion'],
                        logits=model.occlusion))]
                loss_label += ['occlusion head']
            if 'pose' in train_data_dict.keys():
                # c. Pose
                loss_list += [tf.nn.l2_loss(
                    train_data_dict['pose'] - model.pose)]
                loss_label += ['pose head']
                tf.summary.scalar()
            if 'deconv' in config.aux_losses:
                # d. deconvolved image
                loss_list += [tf.nn.l2_loss(
                    model.deconv - train_data_dict['image'])]
                loss_label += ['pose head']
            if 'fc' in config.aux_losses:
                # e. fully convolutional
                fc_shape = [int(x) for x in model.final_fc.get_shape()[1:3]]
                res_images = tf.image.resize_bilinear(train_data_dict['image'], fc_shape)
                # turn background to 0s
                background_mask_value = tf.cast(
                    tf.less(res_images, config.max_depth), tf.float32)
                masked_fc = model.final_fc * background_mask_value
                masked_images = res_images * background_mask_value
                loss_list += [config.fc_lambda * tf.nn.l2_loss(
                    masked_fc - masked_images)]
                loss_label += ['pose head']
                tf.summary.image('FC training activations', model.final_fc)

            loss = tf.add_n(loss_list)

            # Add wd if necessary
            if config.wd_penalty is not None:
                _, l2_wd_layers = tf_fun.fine_tune_prepare_layers(
                    tf.trainable_variables(), config.wd_layers)
                l2_wd_layers = [
                    x for x in l2_wd_layers if 'biases' not in x.name]
                if config.wd_type == 'l1':
                    loss += (config.wd_penalty * tf.add_n(
                            [tf.reduce_sum(tf.abs(x)) for x in l2_wd_layers]))
                elif config.wd_type == 'l2':    
                    loss += (config.wd_penalty * tf.add_n(
                            [tf.nn.l2_loss(x) for x in l2_wd_layers]))

            if config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer
            elif config.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer
            elif config.optimizer == 'momentum':
                #  momentum_var = tf.placeholder(tf.float32, shape=(1))
                optimizer = lambda x: tf.train.MomentumOptimizer(x, momentum=0.1)
            elif config.optimizer == 'rms':
                optimizer = tf.train.RMSPropOptimizer
            else:
                raise 'Unidentified optimizer'

            # Gradient Descent
            optimizer = optimizer(
                config.lr)
            # Op to calculate every variable gradient
            grads = optimizer.compute_gradients(
                loss, tf.trainable_variables())
            # grads = [(tf.clip_by_norm(
            #     g, 8), v) for g, v in grads if g is not None]
            # Op to update all variables according to their gradient
            train_op = optimizer.apply_gradients(
                grads_and_vars=grads)

            # Summarize all gradients and weights
            [tf.summary.histogram(
                var.name + '/gradient', grad)
                for grad, var in grads if grad is not None]
            # train_op = optimizer.minimize(loss)

            # Summarize losses
            [tf.summary.scalar(lab, il) for lab, il in zip(
                loss_label, loss_list)]

            # Summarize images and l1 weights
            tf.summary.image(
                'train images',
                tf.cast(train_data_dict['image'], tf.float32))
            target_filt = [v for v in tf.trainable_variables() if 'conv1_1_filters' in v.name]
            [tf.summary.image(
                f.name,
                tf_fun.put_kernels_on_grid(f)) for f in target_filt]

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                print 'Creating validation graph:'
                val_model = model_file.model_struct()
                val_model.build(
                    rgb=val_data_dict['image'],
                    target_variables=val_data_dict)

                # Calculate validation accuracy
                if 'label' in val_data_dict.keys():
                    val_score = tf.reduce_mean(tf_fun.l2_loss(
                        val_data_dict['label'], val_model.output))
                    tf.summary.scalar("validation mse", val_score)
                if 'fc' in config.aux_losses:
                    tf.summary.image('FC val activations', val_model.final_fc)
                tf.summary.image(
                    'validation images',
                    tf.cast(val_data_dict['image'], tf.float32))

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    tf.add_to_collection('output', model.output)

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create list of variables to run through training model
    train_session_vars = {
        # 'train_op': train_op,
        'loss_value': loss,
        'im': train_data_dict['image'],
        'yhat': model.output,
        'ytrue': train_data_dict['label']
    }
    if hasattr(model, 'deconv'):
        train_session_vars['deconv'] = model.deconv
    if hasattr(model, 'final_fc'):
        train_session_vars['fc'] = model.final_fc

    # Create list of variables to run through validation model
    val_session_vars = {
        'val_acc': val_score,
        'val_pred': val_model.output,
        'val_ims': val_data_dict['image']
    }

    # Start training loop
    np.save(config.train_checkpoint, config)
    step = 0
    num_joints = int(
        train_data_dict['label'].get_shape()[-1]) // config.keep_dims
    normalize_vec = tf_fun.get_normalization_vec(config, num_joints)
    if config.resume_from_checkpoint is not None:
        ckpt = tf.train.latest_checkpoint(config.resume_from_checkpoint)
        print 'Evaluating checkpoint: %s' % ckpt
        saver.restore(sess, ckpt)
    else:
        raise RuntimeError('Set resume_from_checkoint=True in the config')
    try:
        while not coord.should_stop():
            train_out_dict = sess.run(train_session_vars.values())
            train_out_dict = {k: v for k, v in zip(
                train_session_vars.keys(), train_out_dict)}
            assert not np.isnan(
                train_out_dict['loss_value']), 'Model diverged with loss = NaN'

            val_out_dict = sess.run(
                val_session_vars.values())
            val_out_dict = {k: v for k, v in zip(
                val_session_vars.keys(), val_out_dict)}
            if config.normalize_labels:
                train_out_dict['yhat'] *= normalize_vec
                train_out_dict['ytrue'] *= normalize_vec
                val_out_dict['val_pred'] *= normalize_vec
            monkey_mosaic.save_mosaic(
                train_out_dict['im'].squeeze(),
                train_out_dict['yhat'],
                train_out_dict['ytrue'],
                save_fig=False)
            format_str = (
                '%s: step %d | ckpt: %s | validation tf: %s | Train l2 loss = %.8f | '
                'Validation l2 loss = %.8f')
            print (format_str % (
                datetime.now(),
                step,
                ckpt,
                validation_data,
                train_out_dict['loss_value'],
                val_out_dict['val_acc']))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()
        dt_stamp = get_dt()  # date-time stamp
    coord.join(threads)
    sess.close()


def main(validation_data=None, train_data=None, which_joint=None):
    config = monkeyConfig()

    if which_joint is not None:
        config.selected_joints += [which_joint]
    train_and_eval(
        train_data=train_data,
        validation_data=validation_data,
        config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        dest="train_data",
        type=str,
        # default='/home/drew/Desktop/predicted_monkey_on_pole_1/monkey_on_pole.tfrecords',
        help='Train pointer.')
    parser.add_argument(
        "--val",
        dest="validation_data",
        type=str,
        # default='/home/drew/Desktop/predicted_monkey_on_pole_1/monkey_on_pole.tfrecords',
        help='Validation pointer.')
    parser.add_argument(
        "--which_joint",
        dest="which_joint",
        type=str,
        help='Specify a joint to target with the model.')
    args = parser.parse_args()
    main(**vars(args))
