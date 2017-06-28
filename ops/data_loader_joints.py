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


def get_feature_dict(aux_losses):
    feature_dict = {
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    }
    if 'occlusion' in aux_losses:
        feature_dict['occlusion'] = tf.FixedLenFeature([], tf.string)
    return feature_dict


def read_and_decode(
        filename_queue,
        im_size,
        target_size,
        model_input_shape,
        train,
        image_target_size,
        image_input_size,
        maya_conversion,
        max_value,
        normalize_labels,
        joint_names,
        label_shape=22,
        aux_losses=False,
        selected_joints=None,
        background_multiplier=1.01,
        randomize_background=None,
        num_dims=3,
        keep_dims=3,
        clip_z=False,
        mask_occluded_joints=False,
        working_on_kinect=False):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature_dict = get_feature_dict(aux_losses)
    features = tf.parse_single_example(
        serialized_example,
        features=feature_dict
        )
    if max_value is None:
        raise RuntimeError('You must pass a max value')
    # Convert from a scalar string tensor (whose single string has
    label = tf.decode_raw(features['label'], tf.float32)
    image = tf.decode_raw(features['image'], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, np.asarray(target_size))

    # Insert augmentation and preprocessing here
    # image, crop_coors = augment_data(
    #     image, model_input_shape, im_size, train)
    crop_coors = None
    label.set_shape(label_shape)

    if 'convert_labels_to_pixel_space' in train:
        # 1) Resize to config.image_target_size
        # 2) Crop to image size
        label = resize_label_coordinates(
            label,
            image_target_size,
            image_input_size
            )
        if crop_coors is not None:
            label = apply_crop_coordinates(
                label,
                crop_coors
                )

    # Take off first slice of the image
    image = tf.expand_dims(image[:, :, 0], axis=-1)

    if not working_on_kinect:
        print 'Normalizing and adjusting rendered data'
        # Convert background values
        if randomize_background is not None:
            max_value = tf.constant(max_value)
            background_multiplier = tf.constant(
                background_multiplier) + tf.random_uniform(
                (),
                minval=0,
                maxval=randomize_background,
                dtype=tf.float32)

        background_mask = tf.cast(tf.equal(image, 0), tf.float32)
        background_constant = (background_multiplier * max_value)
        background_mask *= background_constant
        image += background_mask

        # Normalize intensity
        image /= background_constant

        # Normalize: must apply max value to image and every 3rd label
        if normalize_labels:
            tile_size = [int(label.get_shape()[0]) / num_dims]

            # Normalize x coor
            lab_adjust = tf.cast(
                tf.tile([image_target_size[0], 1, 1], tile_size), tf.float32)
            label /= lab_adjust

            # Normalize y coor
            lab_adjust = tf.cast(
                tf.tile([1, image_target_size[1], 1], tile_size), tf.float32)
            label /= lab_adjust

            # Normalize z coor
            lab_adjust = tf.cast(
                tf.tile([1, 1, max_value], tile_size), tf.float32)
            label /= lab_adjust
    else:
        print 'Receiving pre-normalized kinect data'

    if clip_z:
        print 'Clipping z dimension'
        # Reshape labels into 2d matrix
        res_size = label_shape // num_dims
        label = tf.reshape(label, [res_size, num_dims])
        split_label = tf.split(label, 3, axis=1)
        label = tf.squeeze(
            tf.reshape(
                tf.stack([split_label[0], split_label[1]], axis=1), [-1, 1]))

    # # Create scatter plot for labels
    # label_scatter = draw_label_coords(
    #     label=label,
    #     canvas_size=[int(x) for x in image.get_shape()[:2]])

    output_data = {
        'label': label,
        'image': image
    }

    if 'occlusion' in aux_losses:
        occlusion = tf.decode_raw(features['occlusion'], tf.float32)
        occlusion.set_shape(label_shape // num_dims)
        output_data['occlusion'] = occlusion

    if 'pose' in aux_losses:
        res_size = label_shape // num_dims
        pose = tf.split(tf.reshape(label, [res_size, num_dims]), res_size)
        neck = pose[1]
        abdomen = pose[2]
        norm_neck = tf.norm(neck)
        norm_abdomen = tf.norm(abdomen)
        output_data['pose'] = tf.acos((
            tf.reduce_sum(neck * abdomen)) / (norm_neck * norm_abdomen))

    if selected_joints is not None:
        assert len(selected_joints) > 0
        res_size = label_shape // num_dims
        split_joints = tf.split(
            tf.reshape(label, [res_size, num_dims]), res_size)
        output_data['label'] = tf.squeeze(
            tf.reshape(
                tf.concat(
                    [split_joints[joint_names.index(j)]
                        for j in selected_joints],
                    axis=0),
                [-1, 1])
            )
        if 'occlusion' in aux_losses:
            split_occlusions = tf.split(
                output_data['occlusion'], res_size)
            output_data['occlusion'] = tf.squeeze(
                tf.concat(
                    [split_occlusions[joint_names.index(
                        j)] for j in selected_joints],
                    axis=0)
                )

    if mask_occluded_joints:
        print 'Masking occluded joints'
        occlusion_masks = tf.reshape(output_data['occlusion'], [-1, 1])
        occlusion_masks = tf.concat(
            [occlusion_masks for x in range(num_dims)], axis=-1)
        occlusion_masks = tf.reshape(occlusion_masks, [-1])
        output_data['label'] = output_data['label'] * tf.cast(
            tf.equal(occlusion_masks, 0),
            tf.float32)  # find non-occluded joints

    if 'z' in aux_losses:
        # Just do h/w
        res_size = label_shape // num_dims
        res_joints = tf.reshape(
                output_data['label'], [res_size, num_dims])
        output_data['z'] = tf.squeeze(
            tf.split(res_joints, num_dims, axis=1)[-1])

    if keep_dims < num_dims:
        print 'Reducing labels from %s to %s dimensions' % (
            num_dims,
            keep_dims)
        res_size = label_shape // num_dims
        split_joints = tf.split(
            tf.reshape(
                label, [res_size, num_dims]),
            num_dims,
            axis=1)[:keep_dims]
        output_data['label'] = tf.reshape(
            tf.concat(
                split_joints,
                axis=1),
            [-1])

    if 'size' in aux_losses:
        # Just do h/w
        res_size = label_shape // num_dims
        res_joints = tf.reshape(
                output_data['label'], [res_size, keep_dims])
        split_joints = tf.split(res_joints, keep_dims, axis=1)
        hw = []
        for s in split_joints:
            hw += [tf.reduce_max(s) - tf.reduce_min(s)]
        output_data['size'] = tf.stack(hw)

    return output_data  # , label_scatter


def draw_label_coords(
        label,
        canvas_size,
        dims=3):
    ls = int(label.get_shape()[0])
    num_el = ls // dims
    label_scatter_coors = tf.reshape(label, [num_el, dims])
    xyzs = tf.split(label_scatter_coors, num_el, axis=0)
    canvas = tf.Variable(tf.zeros(canvas_size))
    canvas_size_tensor = tf.constant(canvas_size)
    uni_off = tf.constant(1)
    for subs in xyzs:
        h = tf.cast(
                tf.reduce_sum(
                    subs * tf.constant([1, 0, 0], dtype=tf.float32)),
                tf.int32)
        w = tf.cast(
                tf.reduce_sum(
                    subs * tf.constant([0, 1, 0], dtype=tf.float32)),
                tf.int32)
        z = tf.reduce_sum(
            subs * tf.constant([0, 0, 1], dtype=tf.float32))
        pre_row_shape = [h - uni_off, canvas_size_tensor[1]]
        post_row_shape = [canvas_size_tensor[0] - (
            h + uni_off), canvas_size_tensor[1]]
        pre_rows = tf.get_variable(pre_row_shape)
        post_rows = tf.zeros(post_row_shape)
        it_row = tf.expand_dims(
            tf.scatter_nd([[w]], [z], [canvas_size[1]]), axis=0)
        new_mat = tf.concat([pre_rows, it_row, post_rows], 0)
        canvas += new_mat
    return canvas


def get_crop_coors(
        image_size,
        target_size):
    h_diff = image_size[0] - target_size[0]
    ts = tf.constant(
        target_size[0], shape=[2, 1])
    offset = tf.cast(
        tf.round(tf.random_uniform([1], minval=0, maxval=h_diff)), tf.int32)
    return offset, ts[0], offset, ts[1]


def apply_crop(
        image,
        target,
        h_min,
        w_min,
        h_max,
        w_max):
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


def slice_op(
        image_slice,
        h_min,
        w_min,
        h_max,
        w_max):
    return tf.slice(
        image_slice, tf.cast(
            tf.concat([h_min, w_min], 0), tf.int32), tf.cast(
            tf.concat([h_max, w_max], 0), tf.int32))


def augment_data(
        image,
        model_input_shape,
        im_size,
        train):
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
            h_min, h_max, w_min, w_max = get_crop_coors(
                image_size=im_size, target_size=model_input_shape)
            image = apply_crop(
                image, model_input_shape, h_min, w_min, h_max, w_max)
            crop_coors = dict()
            crop_coors = {
                'h_min': h_min,
                'h_max': h_max,
                'w_min': w_min,
                'w_max': w_max
             }
            for name in []:
                crop_coors[name] = eval(name)
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
        train=None,
        max_value=None,
        num_epochs=None,
        normalize_labels=True,
        aux_losses=None,
        selected_joints=None,
        joint_names=None,
        mask_occluded_joints=False,
        num_dims=3,
        keep_dims=3,
        background_multiplier=1.01,
        working_on_kinect=False,
        randomize_background=None,
        shuffle=True,
        num_threads=2):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        output_data = read_and_decode(
            filename_queue=filename_queue,
            im_size=im_size,
            target_size=target_size,
            model_input_shape=model_input_shape,
            label_shape=label_shape,
            train=train,
            image_target_size=image_target_size,
            image_input_size=image_input_size,
            maya_conversion=maya_conversion,
            max_value=max_value,
            aux_losses=aux_losses,
            normalize_labels=normalize_labels,
            selected_joints=selected_joints,
            joint_names=joint_names,
            mask_occluded_joints=mask_occluded_joints,
            num_dims=num_dims,
            keep_dims=keep_dims,
            background_multiplier=background_multiplier,
            randomize_background=randomize_background,
            working_on_kinect=working_on_kinect)
        keys = []
        var_list = []
        image = output_data['image']
        var_list += [image]
        keys += ['image']
        label = output_data['label']
        var_list += [label]
        keys += ['label']
        if 'occlusion' in aux_losses:
            occlusion = output_data['occlusion']
            var_list += [occlusion]
            keys += ['occlusion']
        if 'pose' in aux_losses:
            pose = output_data['pose']
            var_list += [pose]
            keys += ['pose']
        if 'z' in aux_losses:
            z = output_data['z']
            var_list += [z]
            keys += ['z']
        if 'size' in aux_losses:
            size = output_data['size']
            var_list += [size]
            keys += ['size']
        if shuffle:
            var_list = tf.train.shuffle_batch(
                var_list,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=1000+3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        else:
            var_list = tf.train.batch(
                var_list,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=1000+3 * batch_size)
        return {k: v for k, v in zip(keys, var_list)}
