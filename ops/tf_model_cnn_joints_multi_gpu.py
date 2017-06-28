import os
import re
import time
import threading
from datetime import datetime
import numpy as np
import tensorflow as tf
import cPickle as pickle
from ops.data_loader_joints import inputs
from ops import tf_fun
from ops import loss_helper
from ops.utils import import_cnn, save_training_data


def training_op(
        iteration,
        gpu,
        scope,
        model_file,
        config,
        train_data_dict):
    with tf.device('/gpu:%s' % gpu):
        if iteration != 0:
            scope.reuse_variables()
        print 'Creating training graph:'
        model = model_file.model_struct(
            weight_npy_path=config.weight_npy_path)
        train_mode = tf.constant(True)  # (name='training', initializer=True)
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
                    config=config,
                    y_key='label',
                    yhat_key='output')
                loss_list += [label_loss]
            else:
                loss_list += tf.nn.l2_loss(
                    model['output'] - train_data_dict['label'])
            loss_label += ['combined head']

        for al in loss_helper.potential_aux_losses():
            loss_list, loss_label = loss_helper.get_aux_losses(
                loss_list=loss_list,
                loss_label=loss_label,
                train_data_dict=train_data_dict,
                model=model,
                aux_loss_dict=al)

        loss = tf.add_n(loss_list)

        # Add wd if necessary
        if config.wd_penalty is not None:
            _, l2_wd_layers = tf_fun.fine_tune_prepare_layers(
                tf.trainable_variables(), config.wd_layers)
            l2_wd_layers = [
                x for x in l2_wd_layers if 'biases' not in x.name]
            if config.wd_type == 'l1':
                loss += (
                    config.wd_penalty * tf.add_n(
                        [tf.reduce_sum(
                            tf.abs(x)) for x in l2_wd_layers]))
            elif config.wd_type == 'l2':
                loss += (
                    config.wd_penalty * tf.add_n(
                        [tf.nn.l2_loss(
                            x) for x in l2_wd_layers]))

        optimizer = loss_helper.return_optimizer(config.optimizer)
        optimizer = optimizer(config.lr)

        if hasattr(model, 'fine_tune_layers'):
            train_op, grads = tf_fun.finetune_learning(
                loss,
                trainables=tf.trainable_variables(),
                fine_tune_layers=model.fine_tune_layers,
                config=config)
        else:
            # Op to calculate every variable gradient
            grads = optimizer.compute_gradients(
                loss, tf.trainable_variables())
            # Op to update all variables according to their gradient
            train_op = optimizer.apply_gradients(
                grads_and_vars=grads)

        # Summarize all gradients and weights
        [tf.summary.histogram(
            '%s/gradient_gpu_%s' % (var.name, gpu), grad)
            for grad, var in grads if grad is not None]
        # train_op = optimizer.minimize(loss)

        # Summarize losses
        [tf.summary.scalar('gpu_%s_%s' % (gpu, lab), il) for lab, il in zip(
            loss_label, loss_list)]

        # Summarize images and l1 weights
        tf.summary.image(
            'train images gpu_%s' % gpu,
            tf.cast(train_data_dict['image'], tf.float32))
        tf_fun.add_filter_summary(
            trainables=tf.trainable_variables(),
            target_layer='conv1_1_filters')
        return train_op, loss, model


def validation_op(
        scope,
        model_file,
        val_data_dict,
        config):
    scope.reuse_variables()
    print 'Creating validation graph:'
    val_model = model_file.model_struct()
    val_model.build(
        rgb=val_data_dict['image'],
        target_variables=val_data_dict)

    # Calculate validation accuracy
    if 'label' in val_data_dict.keys():
        # val_score = tf.nn.l2_loss(
        #     val_model.output - val_data_dict['label'])
        val_score = tf.reduce_mean(
            tf_fun.l2_loss(
                val_model.output,
                val_data_dict['label']))
        tf.summary.scalar("validation mse", val_score)
    if 'fc' in config.aux_losses:
        tf.summary.image(
            'FC val activations', val_model.final_fc)
    tf.summary.image(
        'validation images',
        tf.cast(val_data_dict['image'], tf.float32))


def build_graph(
        selected_gpus,
        model_file,
        train_data_dict,
        config):
    train_list_of_dicts = []
    with tf.variable_scope('cnn') as scope:
        for idx, g in enumerate(selected_gpus):
            train_op, loss, model = training_op(
                iteration=idx,
                gpu=g,
                scope=scope,
                model_file=model_file,
                config=config,
                train_data_dict=train_data_dict)

            # 'train_op': train_op,
            # 'loss_value': loss,
            # 'im': train_data_dict['image'],
            # 'yhat': model.output,
            # 'ytrue': train_data_dict['label']
            train_list_of_dicts += [{
                'train_op': train_op,
                'loss_value': loss,
                'im': train_data_dict['image'],
                'yhat': model.output,
                'ytrue': train_data_dict['label']
            }]
    return train_list_of_dicts


def train_function(
        train_session_vars,
        val_session_vars,
        save_training_vars,
        sess,
        saver,
        coord,
        threads,
        summary_op,
        summary_writer,
        normalize_vec,
        config,
        results_dir):
    try:
        step = 0
        while not coord.should_stop():
            start_time = time.time()
            train_out_dict = sess.run(train_session_vars.values())
            train_out_dict = {k: v for k, v in zip(
                train_session_vars.keys(), train_out_dict)}
            duration = time.time() - start_time
            assert not np.isnan(
                train_out_dict['loss_value']), 'Model diverged with loss = NaN'
            if step % config.steps_before_validation == 0:
                if val_session_vars is not None:
                    val_out_dict = sess.run(
                        val_session_vars.values())
                    val_out_dict = {k: v for k, v in zip(
                        val_session_vars.keys(), val_out_dict)}
                    val_acc = val_out_dict['val_acc']
                    np.savez(
                        os.path.join(
                            results_dir, '%s_val_coors' % step),
                        val_pred=val_out_dict['val_pred'],
                        val_ims=val_out_dict['val_ims'],
                        normalize_vec=normalize_vec)
                else:
                    val_acc = 0.
                with open(
                    os.path.join(
                        results_dir, '%s_config.p' % step), 'wb') as fp:
                    pickle.dump(config, fp)

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy attach 9177
                format_str = (
                    '%s: step %d, loss = %.8f (%.1f examples/sec; '
                    '%.3f sec/batch) | '
                    'Validation l2 loss = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, train_out_dict['loss_value'],
                    config.train_batch / duration, float(duration),
                    val_acc,
                    config.summary_dir))

                # Save the model checkpoint if it's the best yet
                if config.normalize_labels:
                    train_out_dict['yhat'] *= normalize_vec
                    train_out_dict['ytrue'] *= normalize_vec
                [save_training_data(
                    output_dir=results_dir,
                    data=train_out_dict[k],
                    name='%s_%s' % (k, step)) for k in save_training_vars]

                saver.save(
                    sess, os.path.join(
                        config.train_checkpoint,
                        'model_' + str(step) + '.ckpt'), global_step=step)

            else:
                # Training status
                format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; '
                              '%.3f sec/batch)')
                print (format_str % (
                    datetime.now(),
                    step,
                    train_out_dict['loss_value'],
                    config.train_batch / duration,
                    float(duration)))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def train_and_eval(config, selected_gpus=range(2)):
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
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_data_dict = inputs(
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
            normalize_labels=config.normalize_labels,
            aux_losses=config.aux_losses,
            selected_joints=config.selected_joints,
            joint_names=config.joint_order,
            num_dims=config.num_dims,
            keep_dims=config.keep_dims,
            mask_occluded_joints=config.mask_occluded_joints,
            background_multiplier=config.background_multiplier)

        # if config.include_validation:
        #     validation_data = os.path.join(
        #         config.tfrecord_dir,
        #         config.val_tfrecords)
        # else:
        #     validation_data = None

        # val_data_dict = inputs(
        #     tfrecord_file=validation_data,
        #     batch_size=config.validation_batch,
        #     im_size=config.resize,
        #     target_size=config.image_target_size,
        #     model_input_shape=config.resize,
        #     train=config.data_augmentations,
        #     label_shape=config.num_classes,
        #     num_epochs=config.epochs,
        #     image_target_size=config.image_target_size,
        #     image_input_size=config.image_input_size,
        #     maya_conversion=config.maya_conversion,
        #     max_value=config.max_depth,
        #     normalize_labels=config.normalize_labels,
        #     aux_losses=config.aux_losses,
        #     selected_joints=config.selected_joints,
        #     joint_names=config.joint_order,
        #     num_dims=config.num_dims,
        #     keep_dims=config.keep_dims,
        #     mask_occluded_joints=config.mask_occluded_joints,
        #     background_multiplier=config.background_multiplier)

        # Check output_shape
        if config.selected_joints is not None:
            print 'Targeting joint: %s' % config.selected_joints
            joint_shape = len(config.selected_joints) * config.keep_dims
            if (config.num_classes // config.keep_dims) > (joint_shape):
                print 'New target size: %s' % joint_shape
                config.num_classes = joint_shape

    train_session_vars = build_graph(
        selected_gpus,
        model_file,
        train_data_dict,
        config)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(
        config.summary_dir,
        sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(
        sess=sess,
        coord=coord)

    # Create list of variables to save to numpys
    save_training_vars = [
        'im',
        'yhat',
        'ytrue',
        'yhat']

    for al in loss_helper.potential_aux_losses():
        if al.keys()[0] in train_data_dict.keys():
            y_key = '%s' % al.keys()[0]
            train_session_vars[y_key] = al.values()[0]['y_name']
            save_training_vars += [y_key]

            yhat_key = '%s_hat' % al.keys()[0]
            train_session_vars[yhat_key] = al.values()[0]['model_name']
            save_training_vars += [yhat_key]

    # Start training loop
    np.save(config.train_checkpoint, config)
    num_joints = int(
        train_data_dict['label'].get_shape()[-1]) // config.keep_dims
    normalize_vec = tf_fun.get_normalization_vec(config, num_joints)

    # Create multiple threads to run `train_function()` in parallel
    train_threads = []
    for ts in train_session_vars:
        train_args = (
            ts,
            None, # val_session_vars,
            save_training_vars,
            sess,
            saver,
            coord,
            threads,
            summary_op,
            summary_writer,
            normalize_vec,
            config,
            results_dir)
        train_threads.append(
            threading.Thread(
                target=train_function,
                args=train_args))

    # Start the threads, and block on their completion.
    for t in train_threads:
        t.start()
    for t in train_threads:
        t.join()
