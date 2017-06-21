import sys
import numpy as np
import tensorflow as tf
from ops import data_loader_joints
from config import monkeyConfig
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def run_tester(config, file_name):
    print file_name
    filename_queue = tf.train.string_input_producer(
        [file_name],
        num_epochs=None)
    output_data = data_loader_joints.read_and_decode(
        filename_queue=filename_queue,
        im_size=config.resize,
        target_size=config.image_target_size,
        model_input_shape=config.resize,
        label_shape=config.num_classes,
        train=[],
        image_target_size=config.image_target_size,
        image_input_size=config.image_input_size,
        maya_conversion=config.maya_conversion,
        max_value=config.max_depth,
        aux_losses=config.aux_losses,
        normalize_labels=config.normalize_labels,
        selected_joints=config.selected_joints,
        joint_names=config.joint_order,
        mask_occluded_joints=config.mask_occluded_joints,
        num_dims=config.num_dims,
        keep_dims=config.keep_dims,
        background_multiplier=config.background_multiplier
        )

    sess = tf.Session()

    # Required. See below for explanation
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # first example from file
    means = []
    count = 0
    while 1:
        start = timer()
        processed_data = sess.run(output_data)
        import ipdb;ipdb.set_trace()
        means += [np.mean(processed_data['image'])]
        plt.imshow(processed_data['image']); plt.show()
        delta = timer() - start
        sys.stdout.write(
            '\rEntry %s extracted in: %s seconds' % (count, delta)),
        sys.stdout.flush()
        count += 1
        if processed_data['label'] is None:
            sys.stdout.write('No label\n')
            break
        else:
            print processed_data['label']
    np.save('mean_file.npy', means)


if __name__ == '__main__':
    config = monkeyConfig()
    file_name = '/home/drew/Desktop/predicted_monkey_on_pole_1/monkey_on_pole.tfrecords'
    run_tester(
        config=config,
        file_name=file_name)
