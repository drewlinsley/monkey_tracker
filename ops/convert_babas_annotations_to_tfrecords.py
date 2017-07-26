import numpy as np
import os
import shutil
from tqdm import tqdm
from scipy import misc
from ops.data_processing_joints import process_data
from config import monkeyConfig
from ops import tf_fun
from kinect_config import kinectConfig


def main(tmp_folder='tmp'):
    config = monkeyConfig()
    kinect_config = kinectConfig()
    annotations, fnames, flat_ims, start_count = [], [], [], 0
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
                [tf_fun.make_dir(x) for x in [
                    tmp_folder,
                    project_dir,
                    im_folder,
                    label_folder,
                    occlusion_folder]]
            # Images
            ims = np.load(
                os.path.join(
                    it_kinect_config['output_npy_path'], 'frames.npy'))
            ims = ims[data['frame_range']]
            flat_ims += [ims]
            fnames += ['tmp1_%s.npy' % tm for tm in range(
                start_count, start_count + len(ims))]
            start_count = len(ims)
            # Annotations
            for im, it_annotation in tqdm(
                    zip(ims, data['annotations']),
                    total=len(ims),
                    desc='Movie %s' % idx):
                coors = np.zeros((len(config.joint_names), config.num_dims))
                for il, target_joint in enumerate(config.joint_names):
                    for k, v in it_annotation.iteritems():
                        if k == target_joint:
                            coors[il, 0] = v['x']  # v['y']  # This correct??
                            coors[il, 1] = v['y']  # v['x']
                            coors[il, 2] = im[int(v['y']), int(v['x'])]
                annotations += [coors]
        flat_ims = np.concatenate(flat_ims[:])
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

        # Set data folders in config
        config.depth_dir = im_folder
        config.label_dir = label_folder
        config.pixel_label_dir = label_folder
        config.occlusion_dir = occlusion_folder
        config.im_label_dir = im_folder
        config.tfrecord_dir = '/media/data_cifs/monkey_tracking/data_for_babas/tfrecords_from_babas'
        config.use_train_as_val = True
        config.use_image_labels = True
        process_data(config)
        print 'Saved new tfrecords to: %s' % config.tfrecord_dir
    except Exception as e:
        print e
    finally:
        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    main()
