import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ops.utils import get_files
from scipy import misc


def rescale_zo(image):
    mi = np.min(image, keepdims=True).astype(np.float)
    ma = np.max(image, keepdims=True).astype(np.float)
    return (image - mi) / (ma - mi)


def bytes_feature(values):
    """Encodes an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Encodes an int list into a tf int64 list for a tfrecord."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def encode_example(
        im,
        label,
        occlusion=None,
        im_label=None):
    """Encodes a single label/image/feature example into a tfrecord entry."""
    feature = {
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': bytes_feature(label.tostring()),
        'image': bytes_feature(im.tostring()),
    }
    if im_label is not None:
        feature['im_label'] = bytes_feature(im_label.tostring())
    if occlusion is not None:
        feature['occlusion'] = bytes_feature(occlusion.tostring())
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            feature=feature
        )
    )
    return example.SerializeToString()


def cv_files(
        df,
        lf,
        of=None,
        val_prop=0.1):
    """Creates training/validation indices and outputs a dict with
    split depth/label files."""
    rand_idx = np.random.permutation(np.arange(len(df)))
    train_cut = int((1 - val_prop) * len(df))
    train_idx = rand_idx[:train_cut]
    val_idx = rand_idx[train_cut:]
    depth_files = {'train': df[train_idx], 'val': df[val_idx]}
    label_files = {'train': lf[train_idx], 'val': lf[val_idx]}
    if of is not None:
        occlusion_files = {'train': of[train_idx], 'val': of[val_idx]}
    else:
        occlusion_files = {'train': None, 'val': None}
    return depth_files, label_files, occlusion_files


def create_joint_tf_records(
        depth_files,
        label_files,
        tf_file,
        config,
        sample=True,
        occlusions=None):

    """Feature extracts and creates the tfrecords."""
    im_list = []
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i, (depth, label) in tqdm(
                enumerate(
                    zip(depth_files, label_files)), total=len(depth_files)):
            try:
                open(depth)
                open(label)
                if occlusions is not None:
                    open(occlusions[i])
            except:
                continue

            # extract depth image
            depth_image = misc.imread(depth)[:, :, :3]

            # set nans to 0
            depth_image[np.isnan(depth_image)] = 0

            # resize to config.image_target_size if needed
            if config.image_input_size != config.image_target_size:
                depth_image = misc.imresize(
                    depth_image, config.image_target_size[:2])

            # rescale to [0, 1]
            depth_image = rescale_zo(depth_image).astype(np.float32)

            # encode -> tfrecord
            label_vector = np.load(label).astype(np.float32)
            if config.use_image_labels:
                # try:
                im_label = misc.imread(os.path.join(
                    config.im_label_dir, re.split(
                        config.label_extension,
                        re.split('/', label)[-1])[0] + config.image_extension))[:, :, :3]  # label image
                # # If the corresponding label image doesnt exist, skip it
                # except IOError:
                #     continue
                im_label[np.isnan(im_label)] = 0
                if config.image_input_size != config.image_target_size:
                    im_label = misc.imresize(
                        im_label, config.image_target_size[:2])
                im_label = im_label.astype(np.float32)

            im_list.append(np.mean(depth_image))
            if occlusions:
                occlusion = np.load(occlusions[i]).astype(np.float32)
            else:
                occlusion = None
            example = encode_example(depth_image, label_vector, im_label, occlusion)
            tfrecord_writer.write(example)
    return im_list


def extract_depth_features_into_tfrecord(
        depth_files,
        label_files,
        occlusion_files,
        config):
    """Prepares the tf op for nearest neighbor depth features and runs op to
    make tfrecords from them."""

    # Crossvalidate and create tfrecords
    depth_files, label_files, occlusion_files = cv_files(
        depth_files,
        label_files,
        occlusion_files)
    mean_dict = {}
    for k in depth_files.keys():
        print 'Getting depth features: %s' % k
        im_means = create_joint_tf_records(
            depth_files=depth_files[k],
            label_files=label_files[k],
            tf_file=os.path.join(config.tfrecord_dir, '%s.tfrecords' % k),
            config=config,
            sample=config.sample[k]
            occlusions=occlusion_files[k])
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
        os.path.join(
            config.label_dir,
            re.split(
                config.image_extension,
                re.split('/', x)[-1])[0] + config.label_extension)
            for x in depth_files])
    occlusion_files = np.asarray([  # replace the depth dir with label dir
        os.path.join(
            config.occlusion_dir,
            re.split(
                config.image_extension,
                re.split('/', x)[-1])[0] + config.occlusion_extension)
            for x in depth_files])
    if not os.path.isfile(occlusion_files[0]):
        occlusion_files = None
    # Extract directly into a tfrecord
    mean_dict = extract_depth_features_into_tfrecord(
        depth_files=depth_files,
        label_files=label_files,
        occlusion_files=occlusion_files,
        config=config)
    np.savez(
        config.mean_file,
        data=mean_dict,
        **mean_dict)

