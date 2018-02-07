import os
import re
from config import monkeyConfig
from kinect_config import kinectConfig
from glob import glob
from ops import test_tf_kinect
from ops import tf_fun


def main():
    '''Skeleton script for preprocessing and
    passing kinect videos through a trained model'''
    # Find config from the trained model
    videos = kinectConfig()['selected_video']
    if isinstance(videos, list):
        for v in videos:
            process_video(v)
    else:
        process_video(videos)


def process_video(selected_video):
    config = monkeyConfig()
    kinect_config = kinectConfig()
    kinect_config = kinect_config[selected_video]()
    tf_fun.make_dir(kinect_config['data_dir'])
    monkey_files = glob(
        os.path.join(
            kinect_config['kinect_directory'],
            kinect_config['kinect_project'],
            '*%s' % kinect_config['kinect_file_ext']))
    monkey_files = sorted(
        monkey_files, key=lambda name: int(
            re.search('\d+', name.split('/')[-1]).group()))
    test_frames = False

    assert len(monkey_files) > 0, 'Could not find any files!'
    assert kinect_config['output_npy_path'] is not None, 'Need output npy path'

    frames, monkey_files = test_tf_kinect.get_and_trim_frames(
        files=monkey_files,
        start_frame=kinect_config['start_frame'],
        end_frame=kinect_config['end_frame'],
        rotate_frames=kinect_config['rotate_frames'],
        test_frames=test_frames)

    # if kinect_config['use_tfrecords']:
    #     frame_pointer, max_array, frame_toss_index = test_tf_kinect.create_joint_tf_records_for_kinect(
    #         depth_files=frames,
    #         depth_file_names=monkey_files,
    #         model_config=config,
    #         kinect_config=kinect_config)

    # Create preprocessed kinect movie if desired
    if kinect_config['kinect_output_name'] is not None:
        test_tf_kinect.create_movie(
            frames=frames,
            output=kinect_config['kinect_output_name'])

    # Save results to a npz
    files_to_save = {
        'frames': frames,
        'kinect_config': kinect_config,
        'model_config': config,
        # 'frame_toss_index': frame_toss_index,
    }
    test_tf_kinect.save_to_numpys(
        file_dict=files_to_save,
        path=kinect_config['output_npy_path'])


if __name__ == '__main__':
    main()
