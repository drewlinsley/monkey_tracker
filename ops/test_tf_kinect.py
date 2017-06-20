import os
import time
import numpy as np
import tensorflow as tf
import cv2
import itertools
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_opening, binary_closing, \
                                     binary_fill_holes
from matplotlib import pyplot as plt
from matplotlib import animation
from ops.utils import get_dt, import_cnn, save_training_data
from tqdm import tqdm
from datetime import datetime


def save_to_numpys(file_dict, path):
    for k, v in file_dict.iteritems():
        fp = os.path.join(path, k)
        np.save(
            fp,
            v)
        print 'Saved: %s' % fp


def create_movie(frames, output):
    print('Making movie...')
    f, a = plt.subplots(1, 1)
    f.tight_layout()
    artists = [[a.imshow(c)] for c in frames]
    ani = animation.ArtistAnimation(f, artists, interval=50)
    ani.save('%s' % output, 'ffmpeg', 24)


def transform_to_renders(frames, config, rotate=True):
    '''Transform kinect data to the "Maya-space" that
    our renders were produced in.

    inputs::
    frames: Preprocessed kinect data.
    config: Config script from training the joint model.

    outputs::
    frames: Kinect data transformed to maya space'''

    # 1. Trim empty bullshit frames
    frames = np.asarray(frames).astype('float32')
    keep_frames = np.sum(np.sum(frames, axis=-1), axis=-1) > 0
    frames = frames[keep_frames]
    toss_index = np.where(keep_frames)[0]

    # 2. Derive wall position then add it to each frame
    wall_mask = np.asarray([(f == 0).astype(np.float32) for f in frames])
    max_depth_value = np.max(frames)
    wall_position = max_depth_value * config.background_multiplier
    print 'Adding imaginary wall...'
    wall_mask += ((wall_position  - 1) * wall_mask)  # modulate wall position and subtract 1 from masking
    frames += wall_mask  # Because wall_mask is 0s outside of monkey

    # 3. Normalize frames to [0, 1]
    frames /= wall_position

    # 4. Rotate 90 degrees
    frames = np.asarray([np.rot90(f, 3)[:, :, None] for f in frames])

    return frames, toss_index


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


def static_background(frames, left_frame=0, right_frame=40):
    '''
    Combine the left half of `frames[left_frame_no]` with the
    right half of `frames[right_frame_no]`. For good choices
    of `left_frame_no` and `right_frame_no`, this will give
    something like the background of the video.
    '''
    l, r = frames[left_frame], frames[right_frame]
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


def get_and_trim_frames(files, skip_frames_beg, skip_frames_end):
    '''get_and_trim_frames
    Helper for `trim_and_threshold(...)` and `trim_and_bgsub` that
    takes care of loading files from `data_path` and removing
    the first `skip_frames_beg` and last `skip_frames_end` of them.
    '''
    # Get filenames in order
    n = len(files)
    # skip beginning and end
    data = []
    for idx in tqdm(
        range(
            skip_frames_beg, n - skip_frames_end), desc='Trimming frames'):
        data += [np.load(files[idx])]
    return data


def threshold(
        frames,
        low,
        high,
        show_result=False,
        denoise=False,
        remove_objects_smaller_than=48):
    '''threshold
    Zero out values which are greater than `high` or smaller than `low`
    in each frame in `frame`. Do denoising per `denoise` and
    `remove_objects_smaller_than`.
    '''
    # Apply thresholds
    results = [np.zeros_like(f) for f in frames]
    for i, f in tqdm(
            enumerate(frames), desc='Thresholding frames', total=len(frames)):
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
    nframes = len(frames_8bit)
    mask_lists = []
    for i in tqdm(range(wraps), desc='Background subtracting'):
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


def image_batcher(
        start,
        num_batches,
        images,
        batch_size):
    for b in range(num_batches):
        print start, len(images)
        next_image_batch = images[start:start + batch_size]
        image_stack = images[next_image_batch]
        # Add dimensions and concatenate
        yield np.concatenate(
            [x for x in image_stack], axis=0)


def process_kinect_placeholder(model_file, kinect_data, config):
    """Train and evaluate the model."""

    # Import your model
    print 'Model directory: %s' % config.model_output
    print 'Running model: %s' % config.model_type
    model_file = import_cnn(config.model_type)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        val_data_dict = {
            'image': tf.placeholder(
                dtype=tf.float32,
                shape=[None] + config.image_target_size[:2] + [1],
                name='image')
        }

        val_data_dict['image'] = tf.image.resize_image_with_crop_or_pad(
            image=val_data_dict['image'],
            target_height=config.image_target_size[0],
            target_width=config.image_target_size[1])

        # Check output_shape
        if config.selected_joints is not None:
            print 'Targeting joint: %s' % config.selected_joints
            joint_shape = len(config.selected_joints) * config.keep_dims
            if (config.num_classes // config.keep_dims) > (joint_shape):
                print 'New target size: %s' % joint_shape
                config.num_classes = joint_shape

    num_label_targets = config.keep_dims * len(config.joint_order)
    dummy_target = {
        'label': tf.constant(np.zeros(num_label_targets))}
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            print 'Creating training graph:'
            model = model_file.model_struct(
                vgg16_npy_path=config.vgg16_weight_path,
                fine_tune_layers=config.initialize_layers)
            model.build(
                rgb=dummy_target)
            # Model may have multiple "joint prediction" output heads.
            selected_head = model.joint_label_output_keys[0]
            print 'Deriving joint localizations from: %s' % selected_head
            predictions = model[selected_head]

    # Initialize the graph
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))

    # Create list of variables to run through validation model
    val_session_vars = {
        'predictions': predictions,
    }

    yhat_batch = []
    num_joints = len(config.joint_order)
    saver.restore(sess, model_file)
    val_out_dict = sess.run(val_session_vars.values())
    val_out_dict = {k: v for k, v in zip(
        val_session_vars.keys(), val_out_dict)}
    num_batches = len(kinect_data) // config.validation_batch
    for image_batch, file_batch in tqdm(
            image_batcher(
                start=0,
                num_batches=num_batches,
                images=kinect_data,
                batch_size=config.validation_batch),
            total=num_batches):
        feed_dict = {
            val_data_dict['image']: image_batch
        }
        it_yhat = sess.run(
            predictions,
            feed_dict=feed_dict)

        if config.normalize_labels:
            normalize_values = np.asarray(
                config.image_target_size[:2] + [
                    config.max_depth])[:config.keep_dims]
            normalize_vec = normalize_values.reshape(
                1, -1).repeat(num_joints, axis=0).reshape(1, -1)
            it_yhat['yhat'] *= normalize_vec
        yhat_batch += [it_yhat]
    return np.asarray(yhat_batch)
