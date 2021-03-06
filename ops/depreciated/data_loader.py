"""
DEPRECIATED.
"""

import tensorflow as tf
import numpy as np
from scipy import misc
from glob import glob


def get_image_size(config):
    im_size = misc.imread(
      glob(config.train_directory + '*' + config.im_ext)[0]).shape
    if len(im_size) == 2:
        im_size = np.hstack((im_size, 3))
    return im_size


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def repeat_reshape_2d(
        image, im_size, num_channels, tf_dtype=tf.float32,
        img_mean_value=None):
    res_image = tf.reshape(image, np.asarray(im_size)[:num_channels])
    image = tf.cast(repeat_elements(tf.expand_dims(
        res_image, 2), 3, axis=2), tf_dtype)
    if img_mean_value is not None:
        image -= img_mean_value
    return image


def clip_to_value(data, low, high, val, tf_dtype=tf.float32):
    hmask = tf.cast(tf.greater(data, high), tf_dtype)
    lmask = tf.cast(tf.less(data, low), tf_dtype)
    bmask = tf.cast(tf.equal(hmask + lmask, False), tf_dtype)
    return data * bmask


def read_and_decode_single_example(
                    filename, im_size, model_input_shape, num_feats, train,
                    img_mean_value=None, feat_mean_value=None, num_channels=2):
    """first construct a queue containing a list of filenames.
    this lets a user split up there dataset in multiple files to keep
    size down"""
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'feat': tf.FixedLenFeature([], tf.string)
          # flat_shape * 4 (32-bit flaot -> bytes) = 1080000
                }
        )

    # Convert from a scalar string tensor (whose single string has
    label = tf.decode_raw(features['label'], tf.int64)
    image = tf.decode_raw(features['image'], tf.float32)
    feat = tf.decode_raw(features['feat'], tf.float32)

    # Process features specially
    feat = tf.reshape(feat, [im_size[0], im_size[1], num_feats])

    # To support augmentations we have to convert data to 3D
    if num_channels == 2:
        image = repeat_reshape_2d(
            image, im_size, num_channels, img_mean_value=img_mean_value)
    else:
        # Need to reconstruct channels first then transpose channels
        res_image = tf.reshape(image, np.asarray(im_size)[[2, 0, 1]])
        res_image -= img_mean_value
        image = tf.transpose(res_image, [2, 1, 0])

    # Insert augmentation and preprocessing here
    image = augment_data(image, model_input_shape, im_size, train)
    feat = augment_data(feat, model_input_shape, im_size, train)

    # And finally handle the labels
    label = repeat_reshape_2d(
        label, im_size, num_channels, tf_dtype=tf.int32,
        img_mean_value=None)[:, :, 0]

    # Set crazy values to 0
    label = clip_to_value(
        label, np.asarray(-1e4).astype(np.int32),
        np.asarray(1e4).astype(np.int32), 0, tf_dtype=tf.int32)
    image = clip_to_value(
        image, np.asarray(-1e4).astype(np.int32),
        np.asarray(1e4).astype(np.int32), 0)
    feat = clip_to_value(
        feat, np.asarray(-1e4).astype(np.int32),
        np.asarray(1e4).astype(np.int32), 0)

    # Set means
    if img_mean_value is None:
        img_mean_value = 0
    elif img_mean_value == 'within':
        img_mean_value = tf.reduce_mean(image)

    # Set means
    if feat_mean_value is None:
        feat_mean_value = 0
    elif feat_mean_value == 'within':
        feat_mean_value = tf.reduce_mean(feat, axis=[0, 1])
    image -= img_mean_value
    feat -= feat_mean_value
    return label, image, feat


def read_and_decode(
                    filename_queue, im_size, model_input_shape, num_feats,
                    train, img_mean_value=None, feat_mean_value=None,
                    num_channels=2):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'feat': tf.FixedLenFeature([], tf.string)
                }
        )

    # Convert from a scalar string tensor (whose single string has
    label = tf.decode_raw(features['label'], tf.int64)
    image = tf.decode_raw(features['image'], tf.float32)
    feat = tf.decode_raw(features['feat'], tf.float32)

    # Process features specially
    feat = tf.reshape(feat, [im_size[0], im_size[1], num_feats])

    # To support augmentations we have to convert data to 3D
    if num_channels == 2:
        image = repeat_reshape_2d(
            image, im_size, num_channels, img_mean_value=img_mean_value)
    else:
        # Need to reconstruct channels first then transpose channels
        res_image = tf.reshape(image, np.asarray(im_size)[[2, 0, 1]])
        res_image -= img_mean_value
        image = tf.transpose(res_image, [2, 1, 0])

    # Insert augmentation and preprocessing here
    # image = augment_data(image, model_input_shape, im_size, train)
    # feat = augment_data(feat, model_input_shape, im_size, train)

    # And finally handle the labels
    label = repeat_reshape_2d(
        label, im_size, num_channels, tf_dtype=tf.int32,
        img_mean_value=None)[:, :, 0]

    # Set crazy values to 0
    label = clip_to_value(
        label, np.asarray(-1e4).astype(np.int32),
        np.asarray(1e4).astype(np.int32), 0, tf_dtype=tf.int32)
    image = clip_to_value(
        image, np.asarray(-1e4).astype(np.int32),
        np.asarray(1e4).astype(np.int32), 0)
    feat = clip_to_value(
        feat, np.asarray(-1e4).astype(np.int32),
        np.asarray(1e4).astype(np.int32), 0)

    # Set means
    if img_mean_value is None:
        img_mean_value = 0
    elif img_mean_value == 'within':
        img_mean_value = tf.reduce_mean(image)

    # Set means
    if feat_mean_value is None:
        feat_mean_value = 0
    elif feat_mean_value == 'within':
        feat_mean_value = tf.reduce_mean(feat, axis=[0, 1])
    image -= img_mean_value
    feat -= feat_mean_value
    return label, image, feat


def augment_data(image, model_input_shape, im_size, train):
    if train is not None:
        if 'left_right' in train:
            image = tf.image.random_flip_left_right(image)
        if 'up_down' in train:
            image = tf.image.random_flip_up_down(image)
        if 'random_contrast' in train:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        if 'random_brightness' in train:
            image = tf.image.random_brightness(image, max_delta=32./255.)
        if 'rotate' in train:
            image = tf.image.rot90(image, k=np.random.randint(4))
        if 'random_crop' in train:
            image = tf.random_crop(
                image,
                [model_input_shape[0], model_input_shape[1], im_size[2]])
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_shape[0], model_input_shape[1])
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])
    return image


def inputs(
        tfrecord_file, batch_size, im_size, model_input_shape, num_feats,
        train=None, num_epochs=None, feat_mean_value=None, use_features=True):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        label, image, feat = read_and_decode(
            filename_queue=filename_queue,
            im_size=im_size,
            model_input_shape=model_input_shape,
            num_feats=num_feats,
            train=train,
            feat_mean_value=feat_mean_value)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if use_features:
            input_data = feat
        else:
            input_data = image
        data, labels = tf.train.shuffle_batch(
            [input_data, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

        # Finally, have to reshape label -> 1d matrix
        import ipdb;ipdb.set_trace()
        labels = tf.reshape(labels, [batch_size, np.prod(im_size)])
    return data, labels
