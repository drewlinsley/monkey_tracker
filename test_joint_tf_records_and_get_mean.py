import os
import sys
import numpy as np
import tensorflow as tf
from ops import data_loader_joints
from config import monkeyConfig
from timeit import default_timer as timer
from matplotlib import pyplot as plt


def run_tester(config):
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    label, image = data_loader_joints.read_and_decode_single_example(
        filename=train_data,
        im_size=config.resize,
        target_size=config.image_target_size,
        model_input_shape=config.resize,
        train=config.data_augmentations,
        label_shape=config.num_classes)
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
        label_val, image_val = sess.run([label, image])
        means.append(np.mean(image_val))
        import ipdb;ipdb.set_trace()
        plt.imshow(image_val); plt.show()
        delta = timer() - start
        sys.stdout.write(
            '\rEntry %s extracted in: %s seconds' % (count, delta)),
        sys.stdout.flush()
        count += 1
        if label_val is None:
            sys.stdout.write('\n')
            break
    np.save('mean_file.npy', means)


if __name__ == '__main__':
    config = monkeyConfig()
    run_tester(config)
