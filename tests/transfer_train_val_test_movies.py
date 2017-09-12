"""
DEPRECIATED.
"""

import os
import re
from datetime import datetime
import numpy as np
import tensorflow as tf
from ops.data_loader_joints import inputs
from ops import tf_fun, test_tf_kinect
from ops.utils import get_dt, import_cnn
import argparse
from config import monkeyConfig
from visualization import monkey_mosaic
from scipy.stats import linregress
from matplotlib import pyplot as plt


def train_and_eval(
        train_data,
        validation_data,
        config,
        uniform_batch_size=5,
        swap_datasets=False,
        working_on_kinect=False,
        return_coors=True,
        check_stats=False,
        get_kinect_masks=False,
        babas=False):
    if config.resume_from_checkpoint is not None:
        try:
            if config.augment_background == 'background':
                bg = config.augment_background
            else:
                bg = None
            rfc = config.resume_from_checkpoint
            ic = config.include_validation
            print 'Loading saved config: %s' % config.saved_config
            config = np.load(config.saved_config).item()
            config.resume_from_checkpoint = rfc
            config.include_validation = ic
            if not hasattr(config, 'augment_background'):
                config.augment_background = 'constant'
            if not hasattr(config, 'background_folder'):
                config.background_folder = 'backgrounds'
            if bg is not None:
                print 'Overriding saved config to add kinect backgrounds to training.'
                config.augment_background = bg
            results_dir = rfc
            config.epochs = 1
            if 'domain_adaptation' in config.aux_losses:
                config.aux_losses = [x for x in config.aux_losses if 'domain_adaptation' not in x]
            config.babas_tfrecord_dir = None
            # config.max_depth = 2800.
            # config.background_constant = 600.
        except:
            print 'Relying on default config file.'

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
    if uniform_batch_size is None:
        uniform_batch_size = config.train_batch
    print 'Batch size: %s' % uniform_batch_size
    num_epochs = 1
    with tf.device('/cpu:0'):
        print 'Using max of %s on train dataset: %s' % (
            config.max_depth, train_data)

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
            background_multiplier=config.background_multiplier,
            augment_background='constant',
            background_folder=config.background_folder,
            randomize_background=config.randomize_background,
            maya_joint_labels=config.labels,
            # babas_tfrecord_dir=train_babas_tfrecord_dir,
            shuffle=False,
            convert_labels_to_pixel_space=config.convert_labels_to_pixel_space)
        train_data_dict['deconv_label_size'] = len(config.labels)

        val_data_dict = inputs(
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
            normalize_labels=config.normalize_labels,
            aux_losses=config.aux_losses,
            selected_joints=config.selected_joints,
            joint_names=config.joint_order,
            num_dims=config.num_dims,
            keep_dims=config.keep_dims,
            mask_occluded_joints=config.mask_occluded_joints,
            background_multiplier=config.background_multiplier,
            augment_background=config.augment_background,
            background_folder=config.background_folder,
            randomize_background=config.randomize_background,
            maya_joint_labels=config.labels,
            # babas_tfrecord_dir=val_babas_tfrecord_dir,
            shuffle=False,
            convert_labels_to_pixel_space=config.convert_labels_to_pixel_space)
        val_data_dict['deconv_label_size'] = len(config.labels)

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
            model = model_file.model_struct()
            train_mode = tf.get_variable(name='training', initializer=False)
            model.build(
                rgb=train_data_dict['image'],
                target_variables=train_data_dict,
                train_mode=train_mode,
                batchnorm=config.batch_norm)

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                print 'Creating validation graph:'
                val_model = model_file.model_struct()
                val_model.build(
                    rgb=val_data_dict['image'],
                    target_variables=val_data_dict,
                    train_mode=train_mode,
                    batchnorm=[None])

                val_score = tf.reduce_mean(tf_fun.l2_loss(
                    val_data_dict['label'], val_model.output))

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    # tf.add_to_collection('output', model.output)

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
        'yhat': model.output,
        'im': train_data_dict['image'],
        'ytrue': train_data_dict['label'],
    }
    if get_kinect_masks:
        var_a = model.high_feature_encoder_1x1_0
        var_b = tf.image.resize_bilinear(
            model.high_feature_encoder_1x1_1, [int(
                x) for x in model.high_feature_encoder_1x1_0.get_shape()[1:3]])
        var_c = tf.image.resize_bilinear(
            model.high_feature_encoder_1x1_1, [int(
                x) for x in model.high_feature_encoder_1x1_0.get_shape()[1:3]])
        train_session_vars['monkey_mask'] = tf.reduce_mean(
            tf.pow(var_a + var_b + var_c, 2), axis=3)

    # Create list of variables to run through validation model
    val_session_vars = {
        'val_acc': val_score,
        'val_pred': val_model.output,
        'val_ims': val_data_dict['image']
    }

    if hasattr(model, 'deconv'):
        train_session_vars['deconv'] = model.deconv
        val_session_vars['deconv'] = val_model.deconv
    if hasattr(model, 'z'):
        train_session_vars['z'] = model.z
        val_session_vars['z'] = val_model.z

    # Start training loop
    np.save(config.train_checkpoint, config)
    step = 0
    num_joints = int(
        train_data_dict['label'].get_shape()[-1]) // config.keep_dims
    normalize_vec = tf_fun.get_normalization_vec(config, num_joints)
    if babas:
        normalize_vec *= 2
    joint_predictions, joint_gt, out_ims, monkey_masks, joint_zs = [], [], [], [], []
    val_joint_predictions, val_joint_gt, val_out_ims, val_joint_zs = [], [], [], []

    if config.resume_from_checkpoint is not None:
        if '.ckpt' in config.resume_from_checkpoint:
            ckpt = config.resume_from_checkpoint
            saver.restore(sess, ckpt)
        else:
            ckpt = tf.train.latest_checkpoint(config.resume_from_checkpoint)
            print 'Evaluating checkpoint: %s' % ckpt
            saver.restore(sess, ckpt)
    else:
        raise RuntimeError('Set resume_from_checkpoint field in the config.')
    try:
        while not coord.should_stop():
            train_out_dict = sess.run(train_session_vars.values())
            train_out_dict = {k: v for k, v in zip(
                train_session_vars.keys(), train_out_dict)}
            if check_stats:
                slopes, intercepts = [], []
                for yhat, y in zip(
                        train_out_dict['yhat'],
                        train_out_dict['ytrue']):
                    slope, intercept, r_v, p_v, std = linregress(yhat, y)
                    slopes += [slope]
                    intercepts += [intercept]
                plt.hist(slopes, 100)
                plt.show()
                plt.hist(intercepts)
                plt.show()
            val_out_dict = sess.run(
                val_session_vars.values())
            val_out_dict = {k: v for k, v in zip(
                val_session_vars.keys(), val_out_dict)}
            if get_kinect_masks:
                monkey_masks += [train_out_dict['monkey_mask']]
            else:
                if config.normalize_labels:
                    # Postlabel normalization
                    train_out_dict['yhat'] *= normalize_vec
                    train_out_dict['ytrue'] *= normalize_vec
                    val_out_dict['val_pred'] *= normalize_vec
                train_out_dict['yhat'][train_out_dict['yhat'] < 0] = 0
                val_out_dict['val_pred'][val_out_dict['val_pred'] < 0] = 0
                if return_coors:
                    joint_predictions += [train_out_dict['yhat']]
                    joint_zs += [train_out_dict['z']]
                    joint_gt += [train_out_dict['ytrue']]
                    out_ims += [train_out_dict['im'].squeeze()]
                    val_joint_predictions += [val_out_dict['val_pred']]
                    val_joint_zs += [val_out_dict['z']]
                    val_joint_gt += [val_out_dict['val_pred']]
                    val_out_ims += [val_out_dict['val_ims'].squeeze()]
                else:
                    if 'z' in train_out_dict.keys():
                        # Fix this... Not right
                        h, w = train_out_dict['yhat'].shape
                        res_yhat = train_out_dict['yhat'].reshape(h, w // 2, 2)
                        pxs = res_yhat[:, :, 0]
                        pys = res_yhat[:, :, 1]
                        monkey_mosaic.save_3d_mosaic(
                            ims=train_out_dict['im'].squeeze(),
                            pxs=pxs,
                            pys=pys,
                            pzs=val_out_dict['z'] * config.max_depth,
                            ys=None,  # train_out_dict['ytrue'],
                            save_fig=False)
                    if 'z' in val_out_dict.keys():
                        h, w = val_out_dict['val_pred'].shape
                        res_yhat = val_out_dict['val_pred'].reshape(h, w // 2, 2)
                        pxs = res_yhat[:, :, 0]
                        pys = res_yhat[:, :, 1]
                        monkey_mosaic.save_3d_mosaic(
                            ims=val_out_dict['val_ims'].squeeze(),
                            pxs=pxs,
                            pys=pys,
                            pzs=val_out_dict['z'] * config.max_depth,
                            ys=None,  # train_out_dict['ytrue'],
                            save_fig=False)

                    monkey_mosaic.save_mosaic(
                        train_out_dict['im'].squeeze(),
                        train_out_dict['yhat'],
                        None,  # train_out_dict['ytrue'],
                        save_fig=False)
                    monkey_mosaic.save_mosaic(
                        val_out_dict['val_ims'].squeeze(),
                        val_out_dict['val_pred'],
                        None,  # train_out_dict['ytrue'],
                        save_fig=False)

            format_str = (
                '%s: step %d | ckpt: %s | validation tf: %s | Train l2 loss = %.8f | '
                'Validation l2 loss = %.8f')
            print (format_str % (
                datetime.now(),
                step,
                ckpt,
                validation_data,
                0.,
                val_out_dict['val_acc']))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (num_epochs, step))
    finally:
        coord.request_stop()
        dt_stamp = get_dt()  # date-time stamp
    coord.join(threads)
    sess.close()
    tf.reset_default_graph()
    # REMOVE BELOW HERE TO NEXT COMMENT
    # np.savez('test_data', yhat=np.concatenate(joint_predictions).squeeze(), ytrue=np.concatenate(joint_gt).squeeze(), im=np.concatenate(out_ims))

    # joint_dict = {
    #         'yhat': np.concatenate(joint_predictions).squeeze(),
    #         'ytrue': np.concatenate(joint_gt).squeeze(),
    #         'im': np.concatenate(out_ims)
    #         }
    # list_of_yhat_joints = []
    # for yhats in joint_dict['yhat']:
    #     res_yhats = yhats.reshape(-1, 2)
    #     frame_dict = {}
    #     for k, row in zip(config.joint_names, res_yhats):
    #         frame_dict[k] = {'x': float(row[0]), 'y': float(row[1])}
    #     list_of_yhat_joints += [frame_dict]
    # with open('test.json', 'w') as fout:
    #     json.dump(list_of_yhat_joints, fout)
    # print 'JSON saved to: %s' %'test.json'  #  kinect_config['output_json_path']
    
    if return_coors:
        produce_3ds = True
        if produce_3ds:
            xys = np.concatenate(joint_predictions).squeeze()
            h, w = xys.shape
            xys = xys.reshape(h, w // 2, 2)
            pxs = xys[:, :, 0]
            pys = xys[:, :, 1]
            joint_dict = {
                'pxs': pxs,
                'pys': pys,
                'pzs': np.concatenate(joint_zs).squeeze(),
                'im': np.concatenate(out_ims)
                }
            overlaid_pred = test_tf_kinect.overlay_joints_frames_3d(
                joint_dict=joint_dict,
                output_folder='train_pred_ims')
            test_tf_kinect.create_movie(
                files=overlaid_pred,
                output='pole3d_vid.mp4')
            xys = np.concatenate(val_joint_predictions).squeeze()
            h, w = xys.shape
            xys = xys.reshape(h, w // 2, 2)
            pxs = xys[:, :, 0]
            pys = xys[:, :, 1]
            joint_dict = {
                'pxs': pxs,
                'pys': pys,
                'pzs': np.concatenate(val_joint_zs).squeeze(),
                'im': np.concatenate(val_out_ims)
                }
            overlaid_pred = test_tf_kinect.overlay_joints_frames_3d(
                joint_dict=joint_dict,
                output_folder='val_pred_ims')
            test_tf_kinect.create_movie(
                files=overlaid_pred,
                output='cage3d_vid.mp4')

        else:
            joint_dict = {
                'yhat': np.concatenate(joint_predictions).squeeze(),
                'ytrue': np.concatenate(joint_gt).squeeze(),
                'im': np.concatenate(out_ims)
                }
            overlaid_pred = test_tf_kinect.overlay_joints_frames(
                joint_dict=joint_dict,
                output_folder='train_pred_ims')
            test_tf_kinect.create_movie(
                files=overlaid_pred,
                output='train_vid.mp4')
            joint_dict = {
                'yhat': np.concatenate(val_joint_predictions).squeeze(),
                'ytrue': np.concatenate(val_joint_gt).squeeze(),
                'im': np.concatenate(val_out_ims)
                }
            overlaid_pred = test_tf_kinect.overlay_joints_frames(
                joint_dict=joint_dict,
                output_folder='val_pred_ims')
            test_tf_kinect.create_movie(
                files=overlaid_pred,
                output='val_vid.mp4')
            return joint_dict
    elif get_kinect_masks:
        return monkey_masks


def main(
        validation_data=None,
        train_data=None,
        which_joint=None,
        working_on_kinect=True,
        uniform_batch_size=5,
        babas=False):

    config = monkeyConfig()
    if which_joint is not None:
        config.selected_joints += [which_joint]
    train_and_eval(
        train_data=train_data,
        validation_data=validation_data,
        config=config,
        working_on_kinect=working_on_kinect,
        uniform_batch_size=uniform_batch_size,
        babas=babas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        dest="train_data",
        type=str,
        default='/media/data_cifs/monkey_tracking/data_for_babas/tfrecords_from_babas_test/val.tfrecords',  # '/media/data_cifs/monkey_tracking/data_for_babas/tfrecords_from_babas/train.tfrecords',  
        help='Train pointer.')
    parser.add_argument(
        "--val",
        dest="validation_data",
        type=str,
        default='/media/data_cifs/monkey_tracking/data_for_babas/tfrecords_from_babas_test/train.tfrecords',  # '/media/data_cifs/monkey_tracking/data_for_babas/tfrecords_from_babas/train.tfrecords',  
        help='Validation pointer.')
    parser.add_argument(
        "--which_joint",
        dest="which_joint",
        type=str,
        help='Specify a joint to target with the model.')
    parser.add_argument(
        "--kinect",
        dest="working_on_kinect",
        action='store_true',
        help='You are passing kinect data as training.')
    parser.add_argument(
        "--babas",
        dest="babas",
        action='store_true',
        help='You are using babas data.')
    parser.add_argument(
        "--bs",
        dest="uniform_batch_size",
        type=int,
        help='Specify a batch size')
    args = parser.parse_args()
    main(**vars(args))
