"""
DEPRECIATED.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from ops import data_loader_fc
from config import monkeyConfig
from timeit import default_timer as timer
# from matplotlib import pyplot as plt


def run_tester(config):
    train_data = os.path.join(config.tfrecord_dir, 'train.tfrecords')
    label, image, feat = data_loader_fc.read_and_decode_single_example(
        filename=train_data, im_size=config.resize,
        num_feats=config.n_features,
        max_pixels_per_image=config.max_pixels_per_image,
        model_input_shape=config.resize,
        train=[None])

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
        label_val, image_val, features = sess.run([label, image, feat])
        means.append(np.mean(features))
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
