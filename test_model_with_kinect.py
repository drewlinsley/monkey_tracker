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


def main(model_dir, ckpt_name, run_tests=False, reuse_kinect=None, babas=False):
    '''Skeleton script for preprocessing and
    passing kinect videos through a trained model'''
    # Find config from the trained model
    config = monkeyConfig()
    if model_dir is None:
        model_dir = os.path.join(config.model_output, config.model_name)
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

    if reuse_kinect is not None:
        frame_pointer = reuse_kinect
        npz_file = '%s.npz' % frame_pointer.strip('.tfrecords')
        reuse_dict = np.load(npz_file)
        frame_toss_index = reuse_dict['frame_toss_index']
        extents = reuse_dict['extents']
        max_array = reuse_dict['max_array']
        use_kinect = reuse_dict['use_kinect']
        # config.max_depth = max_array
        # config.background_constant = config.max_depth * 2
        joint_dict = train_and_eval(
            frame_pointer,
            frame_pointer,
            config,
            swap_datasets=False,
            working_on_kinect=use_kinect,
            return_coors=True,
            babas=babas)
    else:
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
            frames, extents, frame_toss_index = test_tf_kinect.bb_monkey(frames, kinect_config['time_threshold'])
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

        # Create preprocessed kinect movie if desired
        if kinect_config['kinect_output_name'] is not None:
            tf_fun.make_dir(kinect_config['kinect_output_name']) 
            test_tf_kinect.create_movie(
                frames=frames,
                output=kinect_config['kinect_output_name'])

        import ipdb;ipdb.set_trace()
        # Create tfrecords of kinect data
        if not run_tests:
            # # Transform kinect data to Maya data
            # frames, frame_toss_index = test_tf_kinect.transform_to_renders(
            #     frames=frames,
            #     config=config)
            # use_kinect = True
            # Transform kinect data to Maya data
            frames, frame_toss_index = test_tf_kinect.rescale_kinect_to_maya(
                frames=frames,
                config=config)
            use_kinect = False
        else:
            use_kinect = False

        # Normalize frames
        frames = test_tf_kinect.normalize_frames(
            frames=frames,
            max_value=config.max_depth,
            min_value=config.min_depth,
            max_adj=kinect_config['max_adjust'],
            min_adj=kinect_config['min_adjust'])

        if kinect_config['use_tfrecords']:
            frame_pointer, max_array, it_frame_toss_index = test_tf_kinect.create_joint_tf_records_for_kinect(
                depth_files=frames,
                depth_file_names=monkey_files,
                model_config=config,
                kinect_config=kinect_config)
            frame_toss_index = np.concatenate((frame_toss_index, it_frame_toss_index))
            config.max_depth = max_array
            config.background_constant = config.max_depth * 2
            config.train_batch = 2
            config.val_batch = 2
            joint_dict = train_and_eval(
                frame_pointer,
                frame_pointer,
                config,
                uniform_batch_size=10,
                swap_datasets=False,
                working_on_kinect=use_kinect,
                return_coors=True)
        else:
            # Pass each frame through the CNN
            joint_dict = test_tf_kinect.process_kinect_tensorflow(
                model_ckpt=model_ckpt,
                kinect_data=frames,
                config=config)
            frame_toss_index = []

    # Overlay joint predictions onto frames
    overlaid_pred = test_tf_kinect.overlay_joints_frames(
        joint_dict=joint_dict,
        output_folder=kinect_config['prediction_image_folder'])

    # Create overlay movie if desired
    if kinect_config['predicted_output_name'] is not None:
        test_tf_kinect.create_movie(
            files=overlaid_pred,
            output=kinect_config['predicted_output_name'])

    if len(joint_dict['ytrue']) > 0:
        overlaid_pred = test_tf_kinect.overlay_joints_frames(
            joint_dict=joint_dict,
            output_folder=kinect_config['gt_image_folder'],
            target_key='ytrue')
        # Create overlay movie if desired
        # if kinect_config['gt_output_name'] is not None:
            # test_tf_kinect.create_movie(
                # files=overlaid_gt,
                # output=kinect_config['gt_output_name'])

    # Save results to a npz
    if kinect_config['output_npy_path'] is not None:
        try:
            files_to_save = {
                'overlaid_frames': overlaid_pred,
                'joint_predictions': joint_dict['yhat'],
                'frames': frames,
                'kinect_config': kinect_config,
                'model_config': config,
                'frame_toss_index': frame_toss_index,
                'extents': extents
            }
            test_tf_kinect.save_to_numpys(
                file_dict=files_to_save,
                path=kinect_config['output_npy_path'])
        except:
            print 'Messed up saving results to npz.'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default=None,
        help='Name of model directory.')
    parser.add_argument(
        "--ckpt_name",
        dest="ckpt_name",
        type=str,
        default=None,
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
        help='Passing babas data through.')
    parser.add_argument(
        "--reuse_kinect",
        dest="reuse_kinect",
        type=str,
        default=None,
        help='Pass directory containing processed tf records.')
    args = parser.parse_args()
    main(**vars(args))

