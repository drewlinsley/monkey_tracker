import os
import sys
import numpy as np
import tensorflow as tf
from ops import data_loader_joints
from config import monkeyConfig
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def run_tester(config, file_name):
    if file_name is None:
        file_name = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    print file_name
    filename_queue = tf.train.string_input_producer(
        [file_name],
        num_epochs=1)
    output_data = data_loader_joints.read_and_decode(
        filename_queue=filename_queue,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
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
            working_on_kinect=False,
            augment_background=config.augment_background)

    sess = tf.Session()

    # Required. See below for explanation
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    tf.train.start_queue_runners(sess=sess)

    # first example from file
    means, mins, maxs = [], [], []
    count = 0
    max_count = 10000
    while 1:
        start = timer()
        processed_data = sess.run(output_data)
        means += [np.mean(processed_data['image'])]
        mins += [np.min(processed_data['image'][processed_data['image'] > 0])]
        maxs += [np.max(processed_data['image'])]
        plt.imshow(processed_data['image'].squeeze()); plt.show()
        delta = timer() - start
        sys.stdout.write(
            '\rEntry %s extracted in: %s seconds' % (count, delta)),
        sys.stdout.flush()
        count += 1
        # if processed_data['label'] is None:
        #     sys.stdout.write('No label\n')
        #     break
        # else:
        #     print processed_data['label']
        if count > max_count:
            break
    np.save('mean_file.npy', means)
    np.save('min_file.npy', mins)
    np.save('max_file.npy', maxs)

if __name__ == '__main__':
    config = monkeyConfig()
    # file_name = '/home/drew/Desktop/predicted_monkey_on_pole_1/monkey_on_pole.tfrecords'
    file_name = None
    file_name = '/home/drew/Desktop/predicted_monkey_on_pole_2/monkey_on_pole.tfrecords'
    run_tester(
        config=config,
        file_name=file_name)
