"""
Utility for foreground subtraction of kinect data.

Used to get real-world backgrounds for CNN augmentation.
"""

import os
import numpy as np
from glob import glob
from config import monkeyConfig
from kinect_config import kinectConfig
from ops import test_tf_kinect


def create_background_template(
        frames,
        order_dict,
        rot=-1):
    """Return a thresholded background image."""
    frame_size = np.rot90(np.load(frames[0]).squeeze(), rot).shape
    output = np.zeros((frame_size))
    for k, v in order_dict.iteritems():
        data = np.rot90(np.load(frames[v]).squeeze(), rot)
        if k == 'left':
            output[:, :frame_size[1]//2] = data[:, :frame_size[1]//2]
        elif k == 'top':
            output[:frame_size[0]//2, :frame_size[1]//2] = data[:frame_size[0]//2, :frame_size[1]//2]
        elif k == 'right':
            output[:, -frame_size[1]//2:] = data[:, -frame_size[1]//2:]
        elif k == 'bottom':
            output[-frame_size[0]//2:, -frame_size[1]//2:] = data[-frame_size[0]//2:, -frame_size[1]//2:]
    return output


if __name__ == '__main__':
    """Run the script from config/kinectConfig info."""
    config = monkeyConfig()
    kinect_config = kinectConfig()
    selected_video = kinect_config['selected_video']
    kinect_config = kinect_config[selected_video]()
    target_dir = os.path.join(
        kinect_config['kinect_directory'],
        kinect_config['kinect_project'],
        '*%s' % kinect_config['kinect_file_ext'])
    print 'Searching for files with: %s' % target_dir
    files = glob(target_dir)
    background_template = create_background_template(files, kinect_config['background_mask']) 
    from matplotlib import pyplot as plt
    background_template = test_tf_kinect.crop_aspect_and_resize_center(
        background_template,
        config.image_target_size[:2])
    plt.imshow(background_template);plt.show()
    np.save(os.path.join(config.background_folder, selected_video), background_template)
