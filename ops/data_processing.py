import os
import re
import itertools
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ops.utils import get_files
from ops.feature_extraction import get_depth, create_depth_graph, random_offsets, get_label, clip_df


def bytes_feature(values):
    """Encodes an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Encodes an int list into a tf int64 list for a tfrecord."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def encode_example(im, feat, label):
    """Encodes a single label/image/feature example into a tfrecord entry."""
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': bytes_feature(label.tostring()),
                'image': bytes_feature(im.tostring()),
                'feat': bytes_feature(feat.tostring())
            }
        )
    )
    return example.SerializeToString()


def cv_files(df, lf, val_prop=0.1):
    """Creates training/validation indices and outputs a dict with
    split depth/label files."""
    rand_idx = np.random.permutation(np.arange(len(df)))
    train_cut = int((1 - val_prop) * len(df))
    train_idx = rand_idx[:train_cut]
    val_idx = rand_idx[train_cut:]
    depth_files = {'train': df[train_idx], 'val': df[val_idx]}
    label_files = {'train': lf[train_idx], 'val': lf[val_idx]}
    return depth_files, label_files


def run_depth_feature_extraction(
    sess, dms, depth_image, depth_files,
        label_files, tf_file, config, sample=True):
    """Feature extracts and creates the tfrecords."""
    im_list = []
    feat_list = []
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i, (depth, label) in tqdm(
                enumerate(
                    zip(depth_files, label_files)), total=len(depth_files)):

            # extract depth image
            d = get_depth(
                depth, config.cte_depth, resize=config.resize,
                background_constant=config.background_constant)
            if d is not None:
                feed_dict = {
                    depth_image: d,
                }
                df = clip_df(
                    sess.run(dms, feed_dict), config.background_constant)

                # extract label image
                l = get_label(
                    label, config.background_constant, config.labels,
                    resize=config.resize).ravel()
                idx = np.where(
                    np.logical_and(
                        np.less(
                            l, config.background_constant), np.not_equal(
                            l, 0)))[0]
                if sample is False:
                    it_X = df
                    it_l = l
                else:
                    # Randomly sample from each featureimage
                    idx = np.random.permutation(
                        idx)[:config.max_pixels_per_image]
                    it_X = df.reshape(
                        df.shape[0] * df.shape[1], df.shape[2])[idx, :]
                    it_l = l[idx]
                    if len(it_l) != config.max_pixels_per_image:
                        print 'Label/config pixel length mismatch, skipping: %s' % depth
                    else:
                        im_list.append(np.mean(d))
                        feat_list.append(np.mean(it_X))
                        example = encode_example(d, it_X, it_l)
                        tfrecord_writer.write(example)

            else:
                print 'File has no depth data: %s' % depth
    return feat_list, im_list


def extract_depth_features_into_tfrecord(depth_files, label_files, config):
    """Prepares the tf op for nearest neighbor depth features and runs op to
    make tfrecords from them."""

    # generating the random jumps
    theta = []
    for i in range(config.n_features):  # range(len(depth_files)):
        u = random_offsets(config.offset_nn)
        v = random_offsets(config.offset_nn)
        theta.append([u, v])
    theta = np.asarray(theta)

    # preallocate feature computation matrix
    hw = get_depth(
        depth_files[0], config.cte_depth, resize=config.resize,
        background_constant=config.background_constant).shape
    all_xy = np.asarray(
        list(itertools.product(np.arange(hw[0]), np.arange(hw[1]))))

    # create the feature extraction ops
    with tf.device('/gpu:0'):
        with tf.variable_scope('depth_features'):
            dms, depth_image = create_depth_graph(
                all_xy, theta, hw, config.resize)

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Newer versions of tf may have to replace below with global init
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))

    # Crossvalidate and create tfrecords
    depth_files, label_files = cv_files(depth_files, label_files)
    mean_dict = {}
    for k in depth_files.keys():
        print 'Getting depth features: %s' % k
        feat_means, im_means = run_depth_feature_extraction(
            sess, dms, depth_image, depth_files[k], label_files[k],
            os.path.join(config.tfrecord_dir, '%s.tfrecords' % k),
            config, sample=config.sample[k])
        mean_dict[k + '_feature'] = feat_means
        mean_dict[k + '_image'] = im_means
    print 'Finished'
    return mean_dict


def process_data(config):
    """Extract nearest neighbor features from depth images.
    Eventually move this into the tensorflow graph and handle
    depth files/ label files in batches. Also directly add
    images into a tfrecord instead of into a numpy array."""

    depth_files = get_files(config.depth_dir, config.depth_regex)
    label_files = np.asarray([  # replace the depth dir with label dir
        os.path.join(config.label_dir, re.split('/', x)[-1])
        for x in depth_files])

    # Extract directly into a tfrecord
    mean_dict = extract_depth_features_into_tfrecord(
        depth_files, label_files, config)
    np.savez(config.mean_file, data=mean_dict, **mean_dict)
