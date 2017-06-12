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
    splits = tf.split(axis, x_shape[axis], x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)


def repeat_reshape(
        image, im_size, num_channels=2, tf_dtype=tf.float32,
        img_mean_value=None):
    # res_image = tf.reshape(image, np.asarray([240, 320, 4]))[:,:,:3]
    # image = res_image
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
                    filename, im_size, target_size, model_input_shape, train,
                    label_shape=22):

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
          # flat_shape * 4 (32-bit flaot -> bytes) = 1080000
                }
        )

    # Convert from a scalar string tensor (whose single string has
    label = tf.decode_raw(features['label'], tf.float32)
    image = tf.decode_raw(features['image'], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, np.asarray(target_size))

    # Insert augmentation and preprocessing here
    image = augment_data(image, model_input_shape, im_size, train)
    label.set_shape(label_shape)
    return label, image


def convert_maya_to_pixel(labels, maya_conversion, hw, num_dims=3):
    return ((
        labels * maya_conversion) + tf.cast(tf.tile(  # x/y/z
            np.asarray(
                [hw[0], -1*hw[1], 0]) / 2,
            [int(labels.get_shape()[0]) / num_dims]), tf.float32)) * tf.cast(
        tf.tile([1, -1, 1],
                [int(labels.get_shape()[0]) / num_dims]), tf.float32)


def resize_label_coordinates(
        labels,
        image_target_size,
        image_input_size,
        num_dims=3):
    modifier = np.asarray(
        image_target_size[:2]).astype(np.float32) / np.asarray(
        image_input_size[:2]).astype(np.float32)
    assert modifier[0] == modifier[1]  # Need to generalize eventually
    return labels * tf.cast(
        tf.tile(np.append(modifier, 1), [int(
            labels.get_shape()[0]) / num_dims]), tf.float32)


def flip_lr_coodinates(labels, x_dim):
    return labels * -1 + x_dim


def apply_crop_coordinates(labels, crop_coors, num_dims=3):
    return labels + tf.cast(
        tf.tile([40, -40, 0], [int(
            labels.get_shape()[0]) / num_dims]), tf.float32)


def get_feature_dict(occlusions):
    if occlusions:
        return {
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'occlusion': tf.FixedLenFeature([], tf.string)
                }
    else:
        return {
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
                }


def read_and_decode(
        filename_queue,
        im_size,
        target_size,
        model_input_shape,
        train,
        image_target_size,
        image_input_size,
        maya_conversion,
        label_shape=22,
        occlusions=False):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature_dict = get_feature_dict(occlusions)
    features = tf.parse_single_example(
        serialized_example,
        features=feature_dict
        )

    # Convert from a scalar string tensor (whose single string has
    label = tf.decode_raw(features['label'], tf.float32)
    image = tf.decode_raw(features['image'], tf.float64)
 
    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, np.asarray(target_size))
    image = tf.cast(image, tf.float32)

    # Insert augmentation and preprocessing here
    image, crop_coors = augment_data(image, model_input_shape, im_size, train)
    label.set_shape(label_shape)
    if 'convert_labels_to_pixel_space' in train:
        # 1) Resize to config.image_target_size
        # 2) Crop to image size
        label = resize_label_coordinates(
                    convert_maya_to_pixel(
                        label,
                        maya_conversion,
                        image_input_size
                        ),
                    image_target_size,
                    image_input_size
                    )
        if crop_coors is not None:
            label = apply_crop_coordinates(
                label,
                crop_coors
                )
        elif train is not None:
            adjust = (
                np.asarray(
                    model_input_shape) - np.asarray(
                    image_target_size)) / 2.0
            label = label + tf.cast(
                tf.tile(
                    adjust,
                    [int(label.get_shape()[0]) / len(image_target_size)]),
                tf.float32)
    if occlusions:
        occlusion = tf.decode_raw(features['occlusion'], tf.float32)
        occlusion.set_shape(label_shape // 3)
        print(label.get_shape())
        return label, image, occlusion
    else:
        return label, image


def get_crop_coors(image_size, target_size):
    h_diff = image_size[0] - target_size[0]
    ts = tf.constant(
        target_size[0], shape=[2, 1])
    offset = tf.cast(
        tf.round(tf.random_uniform([1], minval=0, maxval=h_diff)), tf.int32)
    return offset, ts[0], offset, ts[1]


def apply_crop(image, target, h_min, w_min, h_max, w_max):
    im_size = image.get_shape()
    if len(im_size) > 2:
        channels = []
        for idx in range(int(im_size[-1])):
            channels.append(
                slice_op(image[:, :, idx], h_min, w_min, h_max, w_max))
        out_im = tf.stack(channels, axis=2)
        out_im.set_shape([target[0], target[1], int(im_size[-1])])
        return out_im
    else:
        out_im = slice_op(image, h_min, w_min, h_max, w_max)
        return out_im.set_shape([target[0], target[1]])


def slice_op(image_slice, h_min, w_min, h_max, w_max):
    return tf.slice(
        image_slice, tf.cast(
            tf.concat([h_min, w_min], 0), tf.int32), tf.cast(
            tf.concat([h_max, w_max], 0), tf.int32))


def augment_data(image, model_input_shape, im_size, train):
    crop_coors = None
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
            # image = tf.random_crop(
            #     image,
            #     [model_input_shape[0], model_input_shape[1], im_size[2]])
            h_min, h_max, w_min, w_max = get_crop_coors(
                image_size=im_size, target_size=model_input_shape)
            image = apply_crop(
                image, model_input_shape, h_min, w_min, h_max, w_max)
            crop_coors = dict()
            for name in ['h_min', 'h_max', 'w_min', 'w_max']:
                crop_coors[name] = eval(name)
            # crop_coors = dict((
            #     name, eval(name)) for name in [
            #     'h_min', 'h_max', 'w_min', 'w_max'])
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_shape[0], model_input_shape[1])
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])
    return image, crop_coors


def inputs(
        tfrecord_file,
        batch_size,
        im_size,
        target_size,
        model_input_shape,
        label_shape,
        image_target_size,
        image_input_size,
        maya_conversion,
        return_occlusions=None,
        train=None,
        num_epochs=None):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        if return_occlusions is not None:
            label, image, occlusions = read_and_decode(
                filename_queue=filename_queue,
                im_size=im_size,
                target_size=target_size,
                model_input_shape=model_input_shape,
                label_shape=label_shape,
                train=train,
                image_target_size=image_target_size,
                image_input_size=image_input_size,
                maya_conversion=maya_conversion,
                occlusions=True
                )
            data, labels, occlusions = tf.train.shuffle_batch(
                [image, label, occlusions],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000+3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
            return data, labels, occlusions
        else:
            label, image = read_and_decode(
                filename_queue=filename_queue,
                im_size=im_size,
                target_size=target_size,
                model_input_shape=model_input_shape,
                label_shape=label_shape,
                train=train,
                image_target_size=image_target_size,
                image_input_size=image_input_size,
                maya_conversion=maya_conversion
                )

            data, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000+3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
            return data, labels, None
