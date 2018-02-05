"""
Data encoder for tensorflow tfrecords.
"""
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ops.utils import get_files
from scipy import misc


def xyztouvd(x, y, z, focal, cx, cy):
    """Real-world 3d coordinates to image coordinates."""
    v = x/z * focal + cx
    u = -y/z * focal + cy
    d = -z
    return u, v, d


def intrinsic_camera_params(xyz):
    """Convert renders to kinect."""
    # width, height = 512, 424
    # d1, d2 = 1000, 3000
    focal = -365.456
    cx, cy = 256, 212
    x, y, z = xyz
    return xyztouvd(x, y, z, focal, cx, cy)


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
        part=None):
    """Encodes a single label/image/feature example into a tfrecord entry."""
    feature = {
        # A Feature contains one of either a int64_list,
        # float_list, or bytes_list
        'label': bytes_feature(label.tostring()),
        'image': bytes_feature(im.tostring()),
    }
    if part is not None:
        feature['part'] = bytes_feature(part.tostring())
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
        pf=None,
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
    if pf is not None:
        pixel_label_files = {'train': pf[train_idx], 'val': pf[val_idx]}
    else:
        pixel_label_files = {'train': None, 'val': None}
    return depth_files, label_files, occlusion_files, pixel_label_files


def create_joint_tf_records(
        depth_files,
        label_files,
        tf_file,
        config,
        parts=None,
        occlusions=None):
    """Feature extracts and creates the tfrecords."""
    im_list, num_successful = [], 0
    if depth_files[0].split('.')[1] == 'npy':
        use_npy = True
        print 'Getting depth from npys.'
    else:
        use_npy = False
        print 'Getting depth from images.'
    if config.max_train_files is not None:
        num_files = config.max_train_files
    else:
        num_files = len(depth_files)
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i, (depth, label) in tqdm(
                enumerate(
                    zip(depth_files, label_files)),
                total=num_files):

            # extract depth image
            if use_npy:
                depth_image = (np.load(depth)[:, :, :3]).astype(np.float32)
            else:
                depth_image = misc.imread(depth, mode='F')
            depth_image[depth_image == depth_image.min()] = 0.

            # set nans to 0
            depth_image[np.isnan(depth_image)] = 0.
            if depth_image.sum() > 0:  # Because of renders that are all 0

                # encode -> tfrecord
                label_vector = np.asarray(
                    [intrinsic_camera_params(x) for x in np.loadtxt(label)])
                label_vector = label_vector[:, [1, 0, 2]]  # Transpose to h/w\
                label_vector = label_vector[config.lakshmi_order]
                im_list += [np.mean(depth_image)]
                if occlusions is not None:
                    occlusion = np.load(occlusions[i]).astype(np.float32)
                else:
                    occlusion = None
                if parts is not None:
                    part = np.load(parts[i]).astype(np.float32)
                else:
                    part = None
                example = encode_example(
                    im=depth_image.astype(np.float32),
                    label=label_vector,
                    part=part,
                    occlusion=occlusion)
                tfrecord_writer.write(example)
                num_successful += 1
            if (
                config.max_train_files is not None
                    and num_successful > config.max_train):
                break
    return im_list


def extract_depth_features_into_tfrecord(
        depth_files,
        label_files,
        part_files,
        occlusion_files,
        config):
    """Prepares the tf op for nearest neighbor depth features and runs op to
    make tfrecords from them."""

    # Crossvalidate and create tfrecords
    if config.cv_type == 'only_train_data':
        print 'Reusing training data for validation data.'
        depth_files = {'train': depth_files, 'val': depth_files}
        label_files = {'train': label_files, 'val': label_files}
        occlusion_files = {
            'train': occlusion_files, 'val': occlusion_files}
        part_files = {
            'train': part_files, 'val': part_files}
    elif config.cv_type == 'leave_movie_out':
        train_depth = [os.path.join(
            config.depth_dir,
            x) for x in config.cv_inds['train']]
        val_depth = [os.path.join(
            config.depth_dir,
            x) for x in config.cv_inds['val']]
        train_label = [os.path.join(
            config.label_dir,
            x) for x in config.cv_inds['train']]
        val_label = [os.path.join(
            config.label_dir,
            x) for x in config.cv_inds['val']]
        train_occlusion = [os.path.join(
            config.occlusion_dir,
            x) for x in config.cv_inds['train']]
        val_occlusion = [os.path.join(
            config.occlusion_dir,
            x) for x in config.cv_inds['val']]
        train_part_files = [os.path.join(
            config.pixel_label_dir,
            x) for x in config.cv_inds['train']]
        val_part_files = [os.path.join(
            config.pixel_label_dir,
            x) for x in config.cv_inds['val']]
        depth_files = {
            'train': train_depth,
            'val': val_depth
            }
        label_files = {
            'train': train_label,
            'val': val_label
        }
        occlusion_files = {
            'train': train_occlusion,
            'val': val_occlusion
        }
        part_files = {
            'train': train_part_files,
            'val': val_part_files
        }
    else:
        depth_files, label_files, occlusion_files, part_files = cv_files(
            depth_files,
            label_files,
            occlusion_files,
            part_files)
    mean_dict = {}
    for k in depth_files.keys():
        tf_file = os.path.join(
            config.tfrecord_dir,
            '%s_%s.tfrecords' % (config.tfrecord_id, k))
        print 'Building file: %s' % tf_file
        im_means = create_joint_tf_records(
            depth_files=depth_files[k],
            label_files=label_files[k],
            tf_file=tf_file,
            config=config,
            parts=part_files[k],
            occlusions=occlusion_files[k])
        mean_dict[k + '_image'] = im_means
    print 'Finished'
    return mean_dict


def relabel_files(files, new_suffix, new_extension, separator):
    """Split files in f by the path sep and change the suffix to new_suffix."""
    new_f = []
    for f in tqdm(files, total=len(files), desc='Relabeling files'):
        split_path = f.split(os.path.sep)
        path = os.path.sep.join(f.split(os.path.sep)[:-1])
        f_name = split_path[-1]
        proc_file = '%s%s' % (new_suffix, f_name.strip(separator))
        proc_file = '%s.%s' % (proc_file.split('.')[0], new_extension)
        new_f += [os.path.join(path, proc_file)]
    return np.asarray(new_f)


def process_data(config):
    """Extract nearest neighbor features from depth images.
    Eventually move this into the tensorflow graph and handle
    depth files/ label files in batches. Also directly add
    images into a tfrecord instead of into a numpy array."""
    depth_files = get_files(config.render_directory, config.depth_pattern)
    assert len(depth_files) > 0, 'No files found.'
    if config.label_label is not None:
        label_files = relabel_files(
            files=depth_files,
            new_suffix=config.label_label,
            new_extension=config.label_ext,
            separator=config.depth_label)
    else:
        label_files = None
    if config.occlusion_label is not None:
        occlusion_files = relabel_files(
            files=depth_files,
            new_suffix=config.occlusion_label,
            new_extension=config.occlusion_ext,
            separator=config.depth_label)
    else:
        occlusion_files = None
    if config.part_label is not None:
        part_files = relabel_files(
            files=depth_files,
            new_suffix=config.part_label,
            new_extension=config.part_ext,
            separator=config.depth_label)
    else:
        part_files = None

    # Extract directly into a tfrecord
    mean_dict = extract_depth_features_into_tfrecord(
        depth_files=depth_files,
        label_files=label_files,
        part_files=part_files,
        occlusion_files=occlusion_files,
        config=config)
    np.savez(
        os.path.join(config.tfrecord_dir, config.mean_file),
        data=mean_dict,
        **mean_dict)
