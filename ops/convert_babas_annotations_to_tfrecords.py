import numpy as np
from ops.data_processing_joints import process_data
from config import monkeyConfig
from ops import tf_fun
from kinect_config import kinectConfig
import os
import shutil
from tqdm import tqdm



def main(tmp_folder='tmp'):
    config = monkeyConfig()
    kinect_config = kinectConfig()

    # Extract appropriate images and labels
    data = np.load(config.babas_file_for_import)
    project_name = data['video_info'].item()['path'].split(
        '/')[-1].split('.')[0]
    kinect_config = kinect_config[project_name]()
    project_dir = os.path.join(tmp_folder, project_name)
    im_folder = os.path.join(project_dir, 'images')
    label_folder = os.path.join(project_dir, 'labels')
    try:
        [tf_fun.make_dir(x) for x in [tmp_folder, project_dir, im_folder, label_folder]]
        ims = np.load(os.path.join(kinect_config['output_npy_path'], 'frames.npy'))
        ims = ims[data['frame_range']]
        fnames = ['tmp1_%s.npy' % idx for idx in range(len(ims))]
        # Images
        [np.save(
            os.path.join(im_folder, f),
            np.repeat(im[:, :, None], 3, axis=-1)) for f, im in zip(fnames, ims)]
        # Annotations
        annotations = []
        for im, it_annotation in zip(ims, data['annotations']):
            coors = np.zeros((len(config.joint_names), config.num_dims))
            for idx, target_joint in enumerate(config.joint_names):
                for k, v in it_annotation.iteritems():
                    if k == target_joint:
                        coors[idx, 0] = v['x']  # v['y']  # Is this correct??
                        coors[idx, 1] = v['y']  # v['x']
                        coors[idx, 2] = im[v['y'], v['x']]
            annotations += [coors]
        [np.save(os.path.join(label_folder, f), im) for f, im in zip(fnames, annotations)]
        # Set data folders in config
        config.depth_dir = im_folder
        config.label_dir = label_folder
        config.pixel_label_dir = label_folder
        config.occlusion_dir = label_folder
        config.tfrecord_dir = '/media/data_cifs/monkey_tracking/data_for_babas/tfrecords_from_babas'
        config.use_train_as_val = True
        process_data(config)
        print 'Saved new tfrecords to: %s' % config.tfrecord_dir
    except Exception as e:
        print e
    finally:
        shutil.rmtree(tmp_folder)

if __name__ == '__main__':
    main()