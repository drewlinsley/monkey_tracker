import argparse
import os
import re
import json
import cPickle as pickle
import numpy as np
import tensorflow as tf
from config import monkeyConfig
from kinect_config import kinectConfig
from glob import glob
from ops import test_tf_kinect
from ops import utils
from ops import tf_fun
from tests.transfer_train_val_test import train_and_eval


def main(model_dir, ckpt_name, run_tests=False, babas=False):
    '''Skeleton script for preprocessing and
    passing kinect videos through a trained model'''
    # Find config from the trained model
    config = monkeyConfig()
    if config.resume_from_checkpoint is None:
        raise RuntimeError('You must pass a trained checkpoint to this script!')
    if model_dir is None:
        model_dir = os.path.join(config.model_output, config.segmentation_model_name)
    old_resume = config.resume_from_checkpoint
    old_config = config.saved_config
    config.model_name = config.segmentation_model_name
    config.resume_from_checkpoint = os.path.join(
        config.model_output,
        config.model_name)
    config.saved_config = '%s.npy' % config.resume_from_checkpoint
    model_name = model_dir.split('/')[-1]
    config_file = os.path.join(
        config.npy_dir, '%s_config.p' % model_name)
    if os.path.exists(config_file):
        print 'Loading config from trained model.'
        with open(config_file, 'rb') as fp:
            config = pickle.load(fp)
    if ckpt_name is None:
        ckpt_name = tf.train.latest_checkpoint(model_dir)
    model_ckpt = os.path.join(model_dir, ckpt_name)
    kinect_config = kinectConfig()
    kinect_config = kinect_config[kinect_config['selected_video']]()
    tf_fun.make_dir(kinect_config['data_dir'])
    if not run_tests:
        monkey_files = glob(
            os.path.join(
                kinect_config['kinect_directory'],
                kinect_config['kinect_project'],
                '*%s' % kinect_config['kinect_file_ext']))
        monkey_files = sorted(
            monkey_files, key=lambda name: int(
                re.search('\d+', name.split('/')[-1]).group()))
        test_frames = False
    else:
        monkey_files = utils.get_files(config.depth_dir, config.depth_regex)
        kinect_config['start_frame'] = 0
        kinect_config['end_frame'] = 10
        kinect_config['low_threshold'] = None
        kinect_config['high_threshold'] = None
        kinect_config['rotate_frames'] = 0
        kinect_config['run_gmm'] = False
        test_frames = True

    if len(monkey_files) == 0:
        raise RuntimeError('Could not find any files!')

    frames, monkey_files = test_tf_kinect.get_and_trim_frames(
        files=monkey_files,
        start_frame=kinect_config['start_frame'],
        end_frame=kinect_config['end_frame'],
        rotate_frames=kinect_config['rotate_frames'],
        test_frames=test_frames)

    # threshold the depths and do some denoising
    if kinect_config['low_threshold'] is not None and kinect_config['high_threshold'] is not None:
        frames = test_tf_kinect.threshold(
            frames,
            kinect_config['low_threshold'],
            kinect_config['high_threshold'],
            show_result=kinect_config['show_threshold_results'],
            denoise=True)

    # subtract background using MOG and display result
    # this will be very bad, since we're using such a high threshold
    # and doing a lot of openings and closings, but we are only going to
    # use it to estimate a good crop, so that's OK
    if kinect_config['run_gmm']:
        # combine two frames to approximate the background
        bg = test_tf_kinect.static_background(
            frames,
            kinect_config['left_frame'],
            kinect_config['right_frame'])
        frames = [bg] + frames
        frames = test_tf_kinect.bgsub_frames(
            frames,
            kinect_config['bgsub_wraps'],
            kinect_config['bgsub_quorum'],
            show_result=kinect_config['show_mog_result'],
            mog_bg_threshold=kinect_config['bgsub_mog_bg_theshold'])

    if kinect_config['find_bb']:
        frames, extents, frame_toss_index = test_tf_kinect.bb_monkey(
            frames,
            time_threshold=kinect_config['time_threshold'])
    else:
        extents = []
        frame_toss_index = []

    # ignoring the frame generated by `static_background`, estimate
    # a good crop using the result of the MOG
    if kinect_config['crop'] == 'box':
        frames = test_tf_kinect.box_tracking(
            frames[1:],
            kinect_config['w'],
            kinect_config['h'],
            kinect_config['_x'],
            kinect_config['_y'],
            kinect_config['x_'],
            kinect_config['y_'],
            binaries=frames[1:],
            ignore_border_px=kinect_config['ignore_border_px'])
    elif kinect_config['crop'] == 'static':
        frames = [test_tf_kinect.crop_aspect_and_resize_center(
            f, new_size=config.image_target_size[:2]) for f in frames]
    elif kinect_config['crop'] == 'static_and_crop':
        frames = [test_tf_kinect.mask_to_shape(
            f,
            h0=kinect_config['h0'],
            h1=kinect_config['h1'],
            w0=kinect_config['w0'],
            w1=kinect_config['w1']) for f in frames]
        frames = [test_tf_kinect.crop_aspect_and_resize_center(
            f, new_size=config.image_target_size[:2]) for f in frames]

    # # Create tfrecords of kinect data
    frames = np.asarray(frames)
    frame_toss_index = []
    use_kinect = False
    if kinect_config['mask_with_model']:    
        frame_pointer, max_array, it_frame_toss_index = test_tf_kinect.create_joint_tf_records_for_kinect(
            depth_files=frames,
            depth_file_names=monkey_files,
            model_config=config,
            kinect_config=kinect_config)
        image_masks = train_and_eval(
            frame_pointer,
            frame_pointer,
            config,
            uniform_batch_size=None,
            swap_datasets=False,
            working_on_kinect=use_kinect,
            return_coors=False,
            get_kinect_masks=True)        
        all_masks = np.concatenate(image_masks, axis=0)
        num_masks = len(all_masks)
        frames = frames[:num_masks, :, :]
        frames, crop_coors = test_tf_kinect.apply_cnn_masks_to_kinect(
            frames=frames,
            image_masks=all_masks,
            crop_and_pad=kinect_config['crop_and_pad'],
            obj_size=kinect_config['small_object_size'],
            prct=kinect_config['cnn_threshold'])

    # Normalize frames
    frames = test_tf_kinect.normalize_frames(
        frames=frames,
        max_value=config.max_depth,
        min_value=config.min_depth,
        max_adj=kinect_config['max_adjust'],
        min_adj=kinect_config['min_adjust'],
        kinect_max_adj=kinect_config['kinect_max_adjust'],
        kinect_min_adj=kinect_config['kinect_min_adjust'])

    if kinect_config['use_tfrecords']:
        frame_pointer, max_array, it_frame_toss_index = test_tf_kinect.create_joint_tf_records_for_kinect(
            depth_files=frames,
            depth_file_names=monkey_files,
            model_config=config,
            kinect_config=kinect_config)
        frame_toss_index = np.concatenate((frame_toss_index, it_frame_toss_index))
        config.max_depth = max_array
        config.background_constant = config.max_depth * 2
        config.epochs = 1

        if kinect_config['mask_with_model']:
            np.savez(
                '%s' % kinect_config['tfrecord_name'].strip('.tfrecords'),
                frame_toss_index=frame_toss_index,
                extents=extents,
                max_array=max_array,
                use_kinect=use_kinect,
                crop_coors=crop_coors)
            print 'Saved files to %s' % kinect_config['tfrecord_name']
            # Create preprocessed kinect movie if desired
            if kinect_config['kinect_output_name'] is not None:
                test_tf_kinect.create_movie(
                    frames=frames,
                    output=kinect_config['kinect_output_name'],
                    crop_coors=crop_coors)
        print 'Now passing processed frames through selected CNN.'
        config.resume_from_checkpoint = old_resume
        config.saved_config = old_config
        if kinect_config['output_joint_dict']:
            joint_dict = train_and_eval(
                frame_pointer,
                frame_pointer,
                config,
                uniform_batch_size=None,
                swap_datasets=False,
                working_on_kinect=use_kinect,
                return_coors=True,
                babas=babas)
            # Also save json key/value dicts in the same format as BABAS
            list_of_yhat_joints = []
            for yhats in joint_dict['yhat']:
                res_yhats = yhats.reshape(-1, 2)
                frame_dict = {}
                for k, row in zip(config.joint_names, res_yhats):
                    frame_dict[k] = {'x': float(row[0]), 'y': float(row[1])}
                list_of_yhat_joints += [frame_dict]
            with open(kinect_config['output_json_path'], 'w') as fout:
                json.dump(list_of_yhat_joints, fout)
            print 'JSON saved to: %s' % kinect_config['output_json_path']
    else:
        raise RuntimeError('Route is not currently implemented.')
        # Pass each frame through the CNN
        joint_dict = test_tf_kinect.process_kinect_tensorflow(
            model_ckpt=model_ckpt,
            kinect_data=frames,
            config=config)
        frame_toss_index = []

    if kinect_config['output_joint_dict']:
        # Overlay joint predictions onto frames
        tf_fun.make_dir(kinect_config['prediction_image_folder'])
        overlaid_pred = test_tf_kinect.overlay_joints_frames(
            joint_dict=joint_dict,
            output_folder=kinect_config['prediction_image_folder'])

        # Create overlay movie if desired
        if kinect_config['predicted_output_name'] is not None:
            test_tf_kinect.create_movie(
                files=overlaid_pred,
                output=kinect_config['predicted_output_name'])

    # Save results to a npz
    if kinect_config['output_npy_path'] is not None:
        files_to_save = {
            'frames': frames,
            'kinect_config': kinect_config,
            'model_config': config,
            'frame_toss_index': frame_toss_index,
            'extents': extents
        }
        if kinect_config['output_joint_dict']:
            files_to_save['joint_predictions'] = joint_dict['yhat']
            files_to_save['overlaid_frames'] = joint_dict['overlaid_pred']
        test_tf_kinect.save_to_numpys(
            file_dict=files_to_save,
            path=kinect_config['output_npy_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=None,  # '/media/data_cifs/monkey_tracking/results/' + \
            # 'TrueDepth2MilStore/model_output/' + \
            # 'cnn_multiscale_high_res_low_res_skinny_pose_occlusion_2017_06_28_14_21_21',  # 'cnn_multiscale_high_res_low_res_skinny_pose_occlusion_2017_06_27_18_36_53', # 'cnn_multiscale_high_res_low_res_skinny_pose_occlusion_2017_06_23_21_33_30',  # 'cnn_multiscale_high_res_low_res_skinny_pose_occlusion_2017_06_23_10_35_34',
        help='Name of model directory.')
    parser.add_argument(
        "--ckpt_name",
        dest="ckpt_name",
        type=str,
        default=None,  # 'model_49000.ckpt-49000',
        help='Name of TF checkpoint file.')
    parser.add_argument(
        "--test",
        dest="run_tests",
        action='store_true',
        help='Check to see the pipeline works for our renders before transfering to Kinect.')
    parser.add_argument(
        "--babas",
        dest="babas",
        action='store_true',
        help='Babas data.')
    args = parser.parse_args()
    main(**vars(args))