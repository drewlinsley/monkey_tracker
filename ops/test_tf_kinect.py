import os
import re
import numpy as np
import tensorflow as tf
from glob import glob
import cv2
import itertools
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_opening, binary_closing, \
                                     binary_fill_holes
from matplotlib import pyplot as plt
from matplotlib import animation
from ops.utils import get_dt, import_cnn, save_training_data


def denoise_mask(mask):
    return binary_opening(mask.astype(np.bool).astype(np.int))


def mask_voting(tup):
    '''mask_voting
    Helper for `bgsub_frames(...)`. Takes care of denoising
    masks produced by MOG and voting.
    `tup` should contain a frame, a numeric quorum, and a
    tuple of masks.
    '''
    frame, quorum, masks = tup
    r = np.zeros_like(frame)
    m = np.zeros_like(masks[0])
    for mm in masks:
        m[mm.astype(np.bool)] += 1
    # voting
    m = m >= quorum
    binary_opening(m, output=m, iterations=4)
    remove_small_objects(m, min_size=30, in_place=True)
    binary_closing(m, output=m, iterations=4)
    binary_fill_holes(m, output=m)
    m = m.astype(np.bool)
    r[m] = frame[m]
    return r


def static_background(frames, left_frame_no=0, right_frame_no=40):
    '''
    Combine the left half of `frames[left_frame_no]` with the
    right half of `frames[right_frame_no]`. For good choices
    of `left_frame_no` and `right_frame_no`, this will give
    something like the background of the video.
    '''
    l, r = frames[left_frame_no], frames[right_frame_no]
    w = l.shape[0]
    return np.vstack((l[:w/2], r[w/2:]))


def uint16_to_uint8(in_image):
    '''
    Helper to somewhat nicely scale an image from uint16 to uint8 with
    minimal info loss
    '''
    if in_image.dtype == np.uint8:
        return in_image  # just in case
    mn, mx = in_image.min(), in_image.max()
    in_image -= mn
    return np.floor_divide(
        in_image, (mx - mn + 1.) / 256., casting='unsafe').astype(np.uint8)


def get_and_trim_frames(data_path, skip_frames_beg, skip_frames_end):
    '''get_and_trim_frames
    Helper for `trim_and_threshold(...)` and `trim_and_bgsub` that
    takes care of loading files from `data_path` and removing
    the first `skip_frames_beg` and last `skip_frames_end` of them.
    '''
    # Get filenames in order
    print('Getting data...')
    files = sorted(glob(os.path.join(data_path, '*.npy')))
    n = len(files)
    # skip beginning and end
    files = (f for i, f in enumerate(
        files) if (i >= skip_frames_beg and i < n - skip_frames_end))
    # load up frames
    return [np.load(f) for f in files]


def threshold(frames, low, high,
              show_result=False, denoise=False,
              remove_objects_smaller_than=48):
    '''threshold
    Zero out values which are greater than `high` or smaller than `low`
    in each frame in `frame`. Do denoising per `denoise` and
    `remove_objects_smaller_than`.
    '''
    # Apply thresholds
    print('Thresholding...')
    results = [np.zeros_like(f) for f in frames]
    for i, f in enumerate(frames):
        good_ones = (f > low) & (f < high)
        # if denoise, use morphological processing to clean up the mask
        if denoise:
            binary = np.zeros_like(f, dtype=np.int)
            binary[good_ones] = 1
            good_ones = binary_closing(
                binary_opening(binary), iterations=2).astype(np.bool)
        if remove_objects_smaller_than > 0:
            good_ones = remove_small_objects(
                good_ones, min_size=remove_objects_smaller_than)
        results[i][good_ones] = f[good_ones]

    if show_result:
        print('Displaying result...')
        f, (a1, a2) = plt.subplots(
            1, 2, sharex=True, sharey=True, figsize=(10, 5))
        artists = [[a1.imshow(
            o), a2.imshow(r)] for o, r in zip(frames, results)]
        animation.ArtistAnimation(f, artists, interval=50)
        plt.show()
        plt.close('all')

    return results


def trim_and_threshold(
        data_path,
        skip_frames_beg,
        skip_frames_end,
        low,
        high,
        show_result=False,
        denoise=False,
        remove_objects_smaller_than=48):
    '''trim_and_threshold
    Get data and remove start and end frames with `get_and_trim_frames`,
    and then pass all that to `threshold`.
    '''
    frames = get_and_trim_frames(
        data_path,
        skip_frames_beg,
        skip_frames_end)
    return threshold(
        frames,
        low,
        high,
        show_result,
        denoise,
        remove_objects_smaller_than)


def bgsub_frames(
        frames_16bit,
        wraps=32,
        quorum=10,
        show_result=False,
        mog_bg_threshold=2.6):
    '''bgsub_frames
    Given a list of nparrays `frames_16bit`, create `wraps` MOG background
    subtractors with selectivity `mog_bg_threshold` (documented in cv2 docs),
    and have them vote so that a pixel remains in the output if `quorum` of
    them agree that it should.
    '''
    frames_8bit = [uint16_to_uint8(f) for f in frames_16bit]
    # Do the subtraction. Doing a 'wraparound' approach, making masks using
    # video starting at its beginning, and also masks starting from the middle,
    # and then `or`ing the resulting masks to get a result
    print('Subtracting background...')
    nframes = len(frames_8bit)
    mask_lists = []
    for i in range(wraps):
        offset = np.random.randint(nframes)
        fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=50.0)
        offset_frames = frames_8bit[offset:] + frames_8bit[:offset]
        offset_masks = [fgbg.apply(frame) for frame in offset_frames]
        mask_lists.append(offset_masks[-offset:] + offset_masks[:-offset])

    # and, or those masks to mask 16 bit originals
    print('Masking...')
    results_16bit = map(
        mask_voting, zip(
            frames_16bit, itertools.repeat(quorum), zip(*mask_lists)))

    # Display animation if requested
    if show_result:
        print('Displaying result...')
        plt.close('all')
        fig, (a1, a2) = plt.subplots(
            1, 2, sharey=True, sharex=True, figsize=(12, 6))
        artists = [[a1.imshow(f), a2.imshow(r)] for f, r, in zip(
            frames_16bit, results_16bit)]
        a1.set_title('MOG Input')
        a2.set_title('MOG Output')
        animation.ArtistAnimation(fig, artists, interval=50)
        plt.show()
        plt.close('all')

    return results_16bit


def best_box(
        frame,
        w,
        h,
        _x,
        _y,
        x_,
        y_,
        binary=None,
        ignore_border_px=0):
    '''best_box
    In `frame, find the best `w` by `h` window in the shape
    described in `box_tracking(...)`'s docstring. If
    `binary` is provided, find it by finding the window
    with the most nonzero positions in `binary`. else,
    do the same but using `frame` instead of binary.
    `binary` exists as a parameter to allow the user
    to pass in the result of an MOG to get a nice box.
    '''
    # depth doesn't really matter here...
    # we don't want the monkey losing priority to the
    # post in the middle when the monkey is closer to the
    # camera, so clip to [0, 1]
    if binary is None:
        binary = np.clip(frame, 0, 1)
    else:
        np.clip(binary, 0, 1, out=binary)

    # because we're using a greedy strategy, it will pick
    # the first best frame, which tends to have the monkey on
    # an edge. so ignore `ignore_border_px` pixels on the
    # edges to mitigate this
    ww, hh = w - 2 * ignore_border_px, h - 2 * ignore_border_px
    __x, __y = _x + ignore_border_px, _y + ignore_border_px
    x__, y__ = x_ - ignore_border_px, y_ - ignore_border_px

    # results so far
    best, X, Y = 0, 0, 0

    # check sides
    for y in range(__y, y__ - hh):
        if binary[_x:_x + w, y:y + hh].sum() > best:
            best = binary[_x:_x + w, y:y + hh].sum()
            X, Y = _x, y - ignore_border_px
        if binary[x_ - w:x_, y:y + hh].sum() > best:
            best = binary[x_ - w:x_, y:y + hh].sum()
            X, Y = x_ - w, y - ignore_border_px

    # check top and bottom
    for x in range(__x, x__ - ww):
        if binary[x:x + ww, _y:_y + h].sum() > best:
            best = binary[x:x + ww, _y:_y + h].sum()
            X, Y = x - ignore_border_px, _y
        if binary[x:x + ww, y_ - h:y_].sum() > best:
            best = binary[x:x + ww, y_ - h:y_].sum()
            X, Y = x - ignore_border_px, y_ - h
    # print(_x, X, x_ - w, '    ', _y, Y, y_ - h)
    return frame[X:X+w, Y:Y+h]


def box_tracking(
        frames,
        w,
        h,
        _x,
        _y,
        x_,
        y_,
        binaries=None,
        ignore_border_px=0):
    '''
    For each frame, find the `w` by `h` box with the largest sum,
    out of all boxes in the region consisting of the rectangles
    described by the following pairs of opposite corners:
        (_x - w, _y), (_x, y_)
        (x_, _y), (x_ + w, y_)
        (_x, _y - h), (x_, _y)
        (_x, y_), (x_, y_ + h)
    If `binaries` exists, use that to help crop as described
    in `best_box`.
    '''
    print('Box tracking...')
    if binaries is None:
        return [best_box(
            f, w, h, _x, _y, x_, y_, ignore_border_px=ignore_border_px)
                for f in frames]
    else:
        return [best_box(f, w, h, _x, _y, x_, y_, b, ignore_border_px)
                for b, f in zip(binaries, frames)]


def trim_and_bgsub(
        data_path,
        skip_frames_beg,
        skip_frames_end,
        wraps,
        denoise=True,
        show_result=False):
    '''trim_and_bgsub
    Go through the .npy/.npz files at `data_path`. Skip some beginning
    and end frames, and then apply background subtraction to the
    rest. Produce a list of numpy arrays, each of which is a background
    subtracted frame of video. If `show_result`, then will display a
    matplotlib animation so you can check you got the frameskips right.
    Arguments:
        data_path   str     where the data lives. should have .npy files.
        skip_frames_beg, skip_frames_end
                    int     how many frames to cut from the {beg,end} of video
        show_result
                    bool    default False, play the result before returning?
    Returns:
        A list of numpy arrays containing frames of cleaned-up depth video.
    '''
    frames_16bit = get_and_trim_frames(
        data_path,
        skip_frames_beg,
        skip_frames_end)
    return bgsub_frames(
        frames_16bit,
        denoise,
        show_result,
        wraps)


def test_model(model_name, config):
    """Train and evaluate the model."""

    # Import your model
    print 'Model directory: %s' % config.model_output
    print 'Running model: %s' % config.model_type
    model_file = import_cnn(config.model_type)

    # Prepare model training
    dt_stamp = re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    dt_dataset = config.model_type + '_' + dt_stamp + '/'
    if config.selected_joints is not None:
        dt_dataset = '_%s' % (config.selected_joints) + dt_dataset
    config.train_checkpoint = os.path.join(
        config.model_output, dt_dataset)  # timestamp this run
    config.summary_dir = os.path.join(
        config.train_summaries, dt_dataset)
    results_dir = os.path.join(config.npy_dir, dt_stamp)
    print 'Saving Dmurphy\'s online updates to: %s' % results_dir
    dir_list = [config.train_checkpoint, config.summary_dir, results_dir]
    [tf_fun.make_dir(d) for d in dir_list]

    # Prepare model inputs
    train_data = os.path.join(config.tfrecord_dir, config.train_tfrecords)
    if config.include_validation:
        validation_data = os.path.join(config.tfrecord_dir, config.val_tfrecords)
    else:
        validation_data = None

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_data_dict = inputs(
            tfrecord_file=train_data,
            batch_size=config.train_batch,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=config.epochs,
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
            mask_occluded_joints=config.mask_occluded_joints)

        val_data_dict = inputs(
            tfrecord_file=validation_data,
            batch_size=config.validation_batch,
            im_size=config.resize,
            target_size=config.image_target_size,
            model_input_shape=config.resize,
            train=config.data_augmentations,
            label_shape=config.num_classes,
            num_epochs=config.epochs,
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
            mask_occluded_joints=config.mask_occluded_joints)

        # Check output_shape
        if config.selected_joints is not None:
            print 'Targeting joint: %s' % config.selected_joints
            joint_shape = len(config.selected_joints) * config.keep_dims
            if (config.num_classes // config.keep_dims) > (joint_shape):
                print 'New target size: %s' % joint_shape
                config.num_classes = joint_shape

        tf.summary.image(
            'train images',
            tf.cast(train_data_dict['image'], tf.float32))
        tf.summary.image(
            'validation images',
            tf.cast(val_data_dict['image'], tf.float32))

    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            print 'Creating training graph:'
            model = model_file.model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.initialize_layers)
            train_mode = tf.get_variable(name='training', initializer=True)
            model.build(
                rgb=train_data_dict['image'],
                target_variables=train_data_dict,
                train_mode=train_mode,
                batchnorm=config.batch_norm)

            # Prepare the loss functions:::
            loss_list, loss_label = [], []

            if 'label' in train_data_dict.keys():
                # 1. Joint localization loss
                if config.calculate_per_joint_loss:
                    label_loss, use_joints, joint_variance = tf_fun.thomas_l1_loss(
                        model=model,
                        train_data_dict=train_data_dict,
                        config=config)
                    loss_list += [label_loss]
                else:
                    loss_list += [tf.add_n([tf.nn.l2_loss(
                        model[x] - train_data_dict['label']) for x in model.joint_label_output_keys])]
                loss_label += ['combined head']
            if 'occlusion' in train_data_dict.keys():
                # 2. Auxillary losses
                # a. Occlusion
                loss_list += [tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=train_data_dict['occlusion'],
                        logits=model.occlusion))]
                loss_label += ['occlusion head']
            if 'pose' in train_data_dict.keys():
                # c. Pose
                loss_list += [tf.nn.l2_loss(
                    train_data_dict['pose'] - model.pose)]
                loss_label += ['pose head']
                tf.summary.scalar()
            if 'deconv' in config.aux_losses:
                # d. deconvolved image
                loss_list += [tf.nn.l2_loss(
                    model.deconv - train_data_dict['image'])]
                loss_label += ['pose head']
            if 'fc' in config.aux_losses:
                # e. fully convolutional
                fc_shape = [int(x) for x in model.final_fc.get_shape()[1:3]]
                res_images = tf.image.resize_bilinear(train_data_dict['image'], fc_shape)
                # turn background to 0s
                background_mask_value = tf.cast(
                    tf.less(res_images, config.max_depth), tf.float32)
                masked_fc = model.final_fc * background_mask_value
                masked_images = res_images * background_mask_value
                loss_list += [config.fc_lambda * tf.nn.l2_loss(
                    masked_fc - masked_images)]
                loss_label += ['pose head']
                tf.summary.image('FC training activations', model.final_fc)

            loss = tf.add_n(loss_list)

            # Add wd if necessary
            if config.wd_penalty is not None:
                _, l2_wd_layers = tf_fun.fine_tune_prepare_layers(
                    tf.trainable_variables(), config.wd_layers)
                l2_wd_layers = [
                    x for x in l2_wd_layers if 'biases' not in x.name]
                if config.wd_type == 'l1':
                    loss += (config.wd_penalty * tf.add_n(
                            [tf.reduce_sum(tf.abs(x)) for x in l2_wd_layers]))
                elif config.wd_type == 'l2':    
                    loss += (config.wd_penalty * tf.add_n(
                            [tf.nn.l2_loss(x) for x in l2_wd_layers]))

            if config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer
            elif config.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer
            elif config.optimizer == 'momentum':
                #  momentum_var = tf.placeholder(tf.float32, shape=(1))
                optimizer = lambda x: tf.train.MomentumOptimizer(x, momentum=0.1)
            elif config.optimizer == 'rms':
                optimizer = tf.train.RMSPropOptimizer
            else:
                raise 'Unidentified optimizer'

            # Gradient Descent
            optimizer = optimizer(
                config.lr)
            # Op to calculate every variable gradient
            grads = optimizer.compute_gradients(
                loss, tf.trainable_variables())
            # grads = [(tf.clip_by_norm(
            #     g, 8), v) for g, v in grads if g is not None]
            # Op to update all variables according to their gradient
            train_op = optimizer.apply_gradients(
                grads_and_vars=grads)

            # Summarize all gradients and weights
            [tf.summary.histogram(
                var.name + '/gradient', grad)
                for grad, var in grads if grad is not None]
            # train_op = optimizer.minimize(loss)

            # Summarize losses
            [tf.summary.scalar(lab, il) for lab, il in zip(
                loss_label, loss_list)]

            # Setup validation op
            if validation_data is not False:
                scope.reuse_variables()
                print 'Creating validation graph:'
                val_model = model_file.model_struct()
                val_model.build(
                    rgb=val_data_dict['image'],
                    target_variables=val_data_dict)

                # Calculate validation accuracy
                if 'label' in val_data_dict.keys():
                    val_score = tf.nn.l2_loss(
                        val_model.output - val_data_dict['label'])
                    tf.summary.scalar("validation mse", val_score)
                if 'fc' in config.aux_losses:
                    tf.summary.image('FC val activations', val_model.final_fc)


    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create list of variables to run through training model
    train_session_vars = {
        'train_op': train_op,
        'loss_value': loss,
        'im': train_data_dict['image'],
        'yhat': model.output,
        'ytrue': train_data_dict['label']
    }
    if hasattr(model, 'deconv'):
        train_session_vars['deconv'] = model.deconv
    if hasattr(model, 'final_fc'):
        train_session_vars['fc'] = model.final_fc

    # Create list of variables to run through validation model
    val_session_vars = {
        'val_acc': val_score,
        'val_pred': val_model.output,
        'val_ims': val_data_dict['image']
    }

    # Create list of variables to save to numpys
    save_training_vars = [
        'im',
        'yhat',
        'ytrue',
        'yhat'
        ]

    if 'occlusion' in train_data_dict.keys():
        key = 'occhat'
        train_session_vars[key] = model.occlusion,
        save_training_vars += [key]
        key = 'occtrue'
        train_session_vars[key] = train_data_dict['occlusion'],
        save_training_vars += [key]
    if 'pose' in train_data_dict.keys():
        key = 'posehat'
        train_session_vars[key] = model.pose,
        save_training_vars += [key]
        key = 'posetrue'
        train_session_vars[key] = train_data_dict['pose']
        save_training_vars += [key]

    # Start training loop
    np.save(config.train_checkpoint, config)
    step, losses = 0, []
    num_joints = int(
        train_data_dict['label'].get_shape()[-1]) // config.keep_dims
    if config.resume_from_checkpoint is not None:
        print 'Resuming training from checkpoint: %s' % config.resume_from_checkpoint
        saver.restore(sess, config.resume_from_checkpoint)
    try:
        while not coord.should_stop():
            start_time = time.time()
            train_out_dict = sess.run(train_session_vars.values())
            train_out_dict = {k: v for k, v in zip(
                train_session_vars.keys(), train_out_dict)}
            losses.append(train_out_dict['loss_value'])
            duration = time.time() - start_time
            assert not np.isnan(train_out_dict['loss_value']), 'Model diverged with loss = NaN'

            if step % config.steps_before_validation == 0:
                if validation_data is not False:
                    val_out_dict = sess.run(
                        val_session_vars.values())
                    val_out_dict = {k: v for k, v in zip(
                        val_session_vars.keys(), val_out_dict)}
                    np.savez(
                        os.path.join(
                            results_dir, '%s_val_coors' % step),
                        val_pred=val_out_dict['val_pred'],
                        val_ims=val_out_dict['val_ims'],
                        config=config)

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy attach 9177
                format_str = (
                    '%s: step %d, loss = %.4f (%.1f examples/sec; '
                    '%.3f sec/batch) | '
                    'Validation l2 loss = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, train_out_dict['loss_value'],
                    config.train_batch / duration, float(duration),
                    val_out_dict['val_acc'],
                    config.summary_dir))

                # Save the model checkpoint if it's the best yet
                if config.normalize_labels:
                    normalize_values = np.asarray(
                        config.image_target_size[:2] + [
                            config.max_depth])[:config.keep_dims]
                    normalize_vec = normalize_values.reshape(
                        1, -1).repeat(num_joints, axis=0).reshape(1, -1)
                    train_out_dict['yhat'] *= normalize_vec
                    train_out_dict['ytrue'] *= normalize_vec
                [save_training_data(
                    output_dir=results_dir,
                    data=train_out_dict[k],
                    name='%s_%s' % (k, step)) for k in save_training_vars]

                saver.save(
                    sess, os.path.join(
                        config.train_checkpoint,
                        'model_' + str(step) + '.ckpt'), global_step=step)
            else:
                # Training status
                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; '
                              '%.3f sec/batch)')
                print (format_str % (
                    datetime.now(),
                    step,
                    train_out_dict['loss_value'],
                    config.train_batch / duration,
                    float(duration)))
            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (config.epochs, step))
    finally:
        coord.request_stop()

        dt_stamp = get_dt()  # date-time stamp
        np.save(
            os.path.join(
                config.tfrecord_dir, '%straining_loss' % dt_stamp), losses)
    coord.join(threads)
    sess.close()
