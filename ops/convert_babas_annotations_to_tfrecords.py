"""
Script for converting BABAS-annotated frames to tfrecords for CNN training.
"""

import numpy as np
import os
import shutil
from tqdm import tqdm
from ops.data_processing_joints import process_data
from config import monkeyConfig
from ops import tf_fun
from kinect_config import kinectConfig
from ops.test_tf_kinect import overlay_joints_frames
from skimage.transform import resize


def convert_hw_annotations_to_xy(annotations):
    new_annotations = []
    for ann in annotations:
        new_annotations += [ann[:, [1, 0, 2]]]
    return new_annotations


def frame_exclusion():
    """Dictionary of file names/frames to exclude."""
    return {
        # 'babas_monkey_tracking_data_for_babas_processed_videos_monkey_on_pole_3_p1': [
        #     90,
        # ],
        # 'babas_starbuck_pole_new_configuration_competition_depth_0_giovanni_0': [
        #     14,
        # ]

    }


def main(
        tmp_folder='tmp',
        w_pad=-22.,  # -20.,
        h_pad=-10.,
        w_scale=1.125,
        h_scale=1.125,
        debug=True,
        tfrecord_dir='/media/data_cifs/monkey_tracking/data_for_babas/11_1_17_out_of_bag_val',
        fix_image_size=True,
        convert_hw_to_xy=False):
    """Main function for converting BABAS annotated frames to tf records."""
    config = monkeyConfig()
    kinect_config = kinectConfig()
    annotations, fnames, flat_ims, start_count, cv_inds = [], [], [], 0, []
    exclude_frames = frame_exclusion()
    try:
        for idx, batch in enumerate(config.babas_file_for_import):
            # Extract appropriate images and labels
            data = np.load(batch['project'])
            it_kinect_config = kinect_config[batch['data']]()
            if idx == 0:  # just do this once
                project_dir = os.path.join(tmp_folder, batch['data'])
                im_folder = os.path.join(project_dir, 'images')
                label_folder = os.path.join(project_dir, 'labels')
                occlusion_folder = os.path.join(project_dir, 'occlusion')
                debug_folder = os.path.join(project_dir, 'debug')
                [tf_fun.make_dir(x) for x in [
                    tmp_folder,
                    project_dir,
                    im_folder,
                    label_folder,
                    occlusion_folder,
                    debug_folder]]
            # Images
            ims = np.load(
                os.path.join(
                    it_kinect_config['output_npy_path'], 'frames.npy'))
            ims = ims[data['frame_range']]

            # Throw out specified frames
            movie_name = batch['project'].split('/')[-1].split('.')[0]
            if movie_name in exclude_frames.keys():
                keep_frame_list = exclude_frames[movie_name]
                keep_frames = np.asarray(
                    [True if idx not in keep_frame_list else False
                        for x in range(len(ims))])
                ims = ims[keep_frames]
                print 'Trimming to %s ims.' % len(ims)
            else:
                keep_frame_list = [None]

            # Add a CV entry for this movie
            cv_inds += [[batch['cv']] * len(ims)]

            # Fix image size if needed
            if fix_image_size:
                target_size = config.image_target_size[:2]
                if ims[0].shape != target_size:
                    ims = [resize(
                        im,
                        target_size,
                        preserve_range=True) for im in ims]
                    print 'Warning: Image sizes are different than target.' + \
                        ' Converting.'
            flat_ims += [ims]
            fnames += ['tmp1_%s.npy' % tm for tm in range(
                start_count, start_count + len(ims))]
            start_count += len(ims)

            # Annotations
            joint_dict_list = []
            debug_annotations = []
            for idx, (im, it_annotation) in tqdm(
                    enumerate(zip(ims, data['annotations'])),
                    total=len(ims),
                    desc='Movie %s' % idx):
                if idx not in keep_frame_list:
                    it_joint_list = []
                    joint_dict = {}
                    coors = np.zeros(
                        (len(config.joint_names), config.num_dims))
                    for il, target_joint in enumerate(config.joint_names):
                        for k, v in it_annotation.iteritems():
                            if k == target_joint:
                                coors[il, 0] = (v['x'] + w_pad) * w_scale  # (v['x'] * w_scale) + w_pad
                                coors[il, 1] = (v['y'] + h_pad) * h_scale  # (v['y'] * h_scale) + h_pad
                                coors[il, 2] = im[0, 0]  # im[coors[il, 1], coors[il, 2]]
                                it_joint_list += [k]
                    annotations += [coors]
                    debug_annotations += [coors]
                    joint_dict = {
                        k: v[:2] for k, v in zip(it_joint_list, coors)}
                    joint_dict_list += [joint_dict]
                else:
                    print 'Skipping annotation frame %s.' % idx

            if debug:
                # Move frames to a new folder on cifs
                debug_im_dir = os.path.join(
                        tfrecord_dir,
                        '%s_annotated_frame_images' % batch['project'].split(
                            os.path.sep)[-1].split('.')[0])
                tf_fun.make_dir(debug_im_dir)
                joint_dict = {
                    'yhat': debug_annotations,
                    'im': ims
                }
                overlay_joints_frames(
                    joint_dict=joint_dict,
                    output_folder=debug_im_dir,
                    transform_xyz=False
                    )
                print 'Stored annotated images in: %s' % debug_im_dir

        flat_ims = np.concatenate(flat_ims[:])
        if convert_hw_to_xy:
            annotations = convert_hw_annotations_to_xy(annotations)

        [np.save(
            os.path.join(im_folder, f),
            np.repeat(im[:, :, None], 3, axis=-1)) for f, im in zip(
            fnames, flat_ims)]
        [np.save(
            os.path.join(
                label_folder,
                f),
            im) for f, im in zip(
            fnames, annotations)]
        [np.save(
            os.path.join(occlusion_folder, f),
            np.zeros((len(config.joint_names)))) for f in fnames]
        print 'Found %s annotations, %s images.' % (
            len(annotations),
            len(flat_ims))

        # Process cv strings as ints
        flat_cv_inds = np.concatenate(cv_inds)
        train_ind = np.asarray(flat_cv_inds == 'train')
        val_ind = np.asarray(flat_cv_inds == 'validation')

        # Save moments from the annotations
        ann_array = np.asarray(annotations)
        mu = ann_array.mean(0)
        std = ann_array.std(0)
        np.savez(
            os.path.join(
                tfrecord_dir, 'moments'),
            mu=mu,
            std=std)

        # Set data folders in config
        config.depth_dir = im_folder
        config.label_dir = label_folder
        config.pixel_label_dir = label_folder
        config.occlusion_dir = occlusion_folder
        config.im_label_dir = im_folder
        config.tfrecord_dir = tfrecord_dir
        config.special_validation = 'leave_movies_out'
        config.cv_inds = {
            'train': np.asarray(fnames)[train_ind],
            'val': np.asarray(fnames)[val_ind]
        }
        config.use_image_labels = True
        process_data(config)
        print 'Saved new tfrecords to: %s' % config.tfrecord_dir
    except Exception as e:
        print e
    finally:
        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    main()
