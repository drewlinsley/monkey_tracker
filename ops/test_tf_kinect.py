import os
import re
import numpy as np
import tensorflow as tf
import cv2
import itertools
import tf_fun
from scipy import misc
from tqdm import tqdm
from copy import deepcopy
from ops.utils import import_cnn
from ops import data_processing_joints
from ops import data_loader_joints
from visualization import monkey_mosaic
from skimage.transform import resize
from skimage.morphology import remove_small_objects
from skimage import measure
from scipy.ndimage.morphology import binary_opening, binary_closing, \
                                     binary_fill_holes, binary_dilation
import scipy.ndimage.filters as fi
from matplotlib import pyplot as plt
from matplotlib import animation


def apply_cnn_masks_to_kinect(
        frames,
        image_masks,
        prct=85,
        crop_and_pad=True,
        obj_size=500):
    target_shape = frames[0].shape
    of = np.zeros((frames.shape))
    crop_coordinates = []
    print 'Using CNN activities to crop the kinect data.'
    for idx, (f, m) in tqdm(
            enumerate(zip(frames, image_masks)), total=len(frames)):
        m = resize(m, target_shape)
        pm = m > np.percentile(m.ravel(), prct)
        pm = binary_dilation(pm)
        pm = binary_fill_holes(pm)
        pm = binary_fill_holes(pm, structure=np.ones((5,5)))
        tf = f * pm
        if crop_and_pad:
            # Crop keeping the aspect ratio
            rp = measure.regionprops(measure.label(pm))
            areas = np.argsort([r.area for r in rp])[::-1]
            min_row, min_col, max_row, max_col = rp[areas[0]].bbox
            cropped_im = tf[min_row:max_row, min_col:max_col]
            # Center and pad to the appropriate size
            canvas = np.zeros((target_shape))
            cropped_shape = cropped_im.shape
            if cropped_shape[0] % 2:  # odd number
                cropped_im = np.concatenate((
                    cropped_im, np.zeros((1, cropped_shape[1]))), axis=0)
                cropped_shape = cropped_im.shape
            if cropped_shape[1] % 2:  # odd number
                cropped_im = np.concatenate((
                    cropped_im, np.zeros((cropped_shape[0], 1))), axis=1)
                cropped_shape = cropped_im.shape
            min_row = (target_shape[0] // 2) - (cropped_shape[0] // 2)
            min_col = (target_shape[1] // 2) - (cropped_shape[1] // 2)
            max_row = (target_shape[0] // 2) + (cropped_shape[0] // 2)
            max_col = (target_shape[1] // 2) + (cropped_shape[1] // 2)
            canvas[min_row:max_row, min_col:max_col] = cropped_im
            tf = canvas
            crop_coordinates += [min_row, min_col, max_row, max_col]
        # Remove debris
        stuff_mask = tf > 0
        remove_small_objects(stuff_mask, min_size=obj_size, in_place=True)
        tf *= stuff_mask
        of[idx, :, :] = tf
    return of, crop_coordinates


def gkern2(img, kern=3):
    """Returns a 2D Gaussian kernel array."""
    kernel = np.ones((kern,kern),np.float32)/kern**2
    return cv2.filter2D(img,-1,kernel)


def normalize_frames(
        frames,
        max_value,
        min_value,
        min_max_norm=True,
        max_adj=1,
        min_adj=1,
        kinect_max_adj=1,
        kinect_min_adj=1,
        smooth=False):
    """ Need to transform kinect -> maya.
    To do this, for each frame, ID the foreground, then transform its depth to
    maya space."""
    print 'Adjusting max with: %s. Min with: %s.' % (max_adj, min_adj)
    max_value = max_value * max_adj
    min_value = min_value * min_adj
    delta = max_value - min_value
    kinect_max = frames.max() * kinect_max_adj
    kinect_min = frames[frames > 0].min() * kinect_min_adj
    trans_f = np.zeros((frames.shape))
    print 'Normalizing kinect to maya.'
    for idx, f in tqdm(enumerate(frames), total=len(frames)):
        # Normalize foreground to [0, 1]
        zeros_mask = (f > 0).astype(np.float32)
        if min_max_norm:
            normalized_f = (f - kinect_min) / (kinect_max - kinect_min)
            # Convert to maya
            it_f = (normalized_f * (delta)) + min_value
        else:
            normalized_f = f / kinect_max
            it_f = (normalized_f * max_value)
        if smooth:
            it_f = gkern2(it_f)
        it_f *= zeros_mask
        trans_f[idx, :, :] = np.abs(it_f)
    return trans_f


def save_to_numpys(file_dict, path):
    for k, v in file_dict.iteritems():
        fp = os.path.join(path, k)
        np.save(
            fp,
            v)
        print 'Saved: %s' % fp


def create_movie(
        frames=None,
        output=None,
        files=None,
        framerate=30,
        crop_coors=None):
    print('Making movie...')
    if files is not None:
        frames = []
        frames = np.asarray([misc.imread(f) for f in files])
    f, a = plt.subplots(1, 1)
    f.tight_layout()
    artists = []
    plt.axis('off')
    a.set_xticklabels([])
    a.set_yticklabels([])
    f.set_dpi(100)
    DPI = f.get_dpi()
    h, w = frames[0].shape
    f.set_size_inches(w/float(DPI), h/float(DPI))
    for fr in frames:
        it_art = a.imshow(fr, cmap='Greys')
        artists += [[it_art]]
    ani = animation.ArtistAnimation(f, artists, interval=50)
    ani.save(
        output,
        writer='ffmpeg',
        fps=framerate,
        extra_args=['-vcodec', 'h264'])
    # artists = [[a.imshow(fr)] for fr in frames]
    # ani = animation.ArtistAnimation(f, artists, interval=50)
    # ani.save(output, 'ffmpeg', framerate)


def drop_empty_frames(frames, list_type=False):
    if list_type:
        frame_sum = [np.sum(f) for f in frames]
        frames = [f for f, s in zip(frames, frame_sum) if s != 0]
        toss_index = np.where(np.asarray(frame_sum) > 0)[0]
    else:
        frames = np.asarray(frames).astype('float32')
        keep_frames = np.sum(np.sum(frames, axis=-1), axis=-1) > 0
        frames = frames[keep_frames]
        toss_index = np.where(keep_frames)[0]
    return frames, toss_index


def transform_to_renders(
        frames,
        config,
        max_depth_value=None,
        rotate=True,
        pad=False):
    '''Transform kinect data to the "Maya-space" that
    our renders were produced in.

    inputs::
    frames: Preprocessed kinect data.
    config: Config script from training the joint model.

    outputs::
    frames: Kinect data transformed to maya space'''

    # 0. Trim empty bullshit frames
    frames, toss_index = drop_empty_frames(frames)
    if pad:
        padded_frames = []
        for f in frames:
            pre_size = f.shape
            it_frame = np.pad(
                f, [int(x // 4) for x in f.shape],
                'constant',
                constant_values=(0, 0))
            it_frame = resize(
                it_frame,
                pre_size,
                preserve_range=True,
                order=0)
            padded_frames += [it_frame]
        frames = padded_frames

    # 2. Derive wall position then add it to each frame
    wall_mask = np.asarray([(f == 0).astype(np.float32) for f in frames])
    if max_depth_value is None:
        max_depth_value = np.max(frames)
    wall_position = max_depth_value * config.background_multiplier
    print 'Adding imaginary wall...'

    # modulate wall position and subtract 1 from masking
    wall_mask += ((wall_position - 1) * wall_mask)
    frames += wall_mask  # Because wall_mask is 0s outside of monkey

    # 3. Normalize frames to [0, 1]
    frames /= wall_position

    # 4. Add new dimension
    frames = frames[:, :, :, None]

    return frames, toss_index


def rescale_kinect_to_maya(
        frames,
        config):
    frames, toss_index = drop_empty_frames(frames)
    max_depth = np.max(frames)
    frames = (frames / (max_depth * 2)) * config.max_depth  # rescaled
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


def get_and_trim_frames(
        files,
        start_frame,
        end_frame,
        rotate_frames,
        test_frames):
    '''get_and_trim_frames
    Helper for `trim_and_threshold(...)` and `trim_and_bgsub` that
    takes care of loading files from `data_path` and removing
    the first `skip_frames_beg` and last `skip_frames_end` of them.
    '''
    if test_frames:
        files = np.asarray(files)
        files = files[start_frame:end_frame]
        data = [np.load(f).squeeze() for f in files]  # squeeze singletons
        if len(data[0].shape) > 2 and data[0].shape[-1] > 3:
            print 'Detected > 2d images. Trimming excess dimensions.'
        data = [f[:, :, :3] for f in data]
        for idx, d in enumerate(data):
            d[d == d.min()] = 0.
            d[np.isnan(d)] = 0.
            data[idx] = d.astype(np.float32)

        if rotate_frames != 0:
            return np.asarray([np.rot90(d, -1) for d in data]), files
        else:
            return np.asarray(data), files
    else:
        # Get filenames in order
        n = len(files)
        # skip beginning and end
        data = []
        for idx in tqdm(
            range(
                start_frame, n - end_frame), desc='Trimming frames'):
            data += [np.rot90(np.load(files[idx]), rotate_frames)]
        return data, files


def threshold(
        frames,
        low,
        high,
        show_result=False,
        denoise=False,
        remove_objects_smaller_than=400):
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
        proc_im = good_ones * f
        if denoise:
            good_ones = binary_dilation(
                binary_closing(
                    binary_opening(good_ones), iterations=2),
                iterations=2)
        if remove_objects_smaller_than > 0:
            good_ones = remove_small_objects(
                good_ones, min_size=remove_objects_smaller_than)
        results[i] = proc_im * good_ones

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


def bb_monkey(
        frames,
        max_sigma=100,
        threshold=.1,
        denoise=True,
        time_threshold=95,
        crop_image=False):
    # Find "movement mask"
    print 'Deriving movement mask.'
    mm = np.sum(np.asarray(frames), axis=0).astype(np.float32)

    # Threshold at median
    mm[mm > np.percentile(mm[mm > 0], time_threshold)] = 0
    mm = mm > 0

    # Prepare bb stuff
    f_size = frames[0].shape
    crop_frames = []
    bb_size = [f_size[0] / 4, f_size[1] / 4]
    xye = []
    for f in tqdm(frames, total=len(frames)):
        f *= mm  # Mask the image w/ movement
        if crop_image:
            rprops = measure.regionprops(f)
            areas = np.argsort([r.area for r in rprops])[::-1]
            x0, y0 = rprops[areas[0]].centroid
            x0 = int(x0)
            y0 = int(y0)
            minr = np.min([0, x0 - bb_size[0]])
            minc = np.min([0, x0 - bb_size[1]])
            maxr = np.max([f_size[0], x0 + bb_size[0]])
            maxc = np.max([f_size[1], x0 + bb_size[1]])
            f = f[minr:maxr, minc:maxc]
            if denoise:
                good_ones = f > 0
                good_ones = binary_dilation(
                    binary_closing(
                        binary_opening(good_ones), iterations=2),
                    iterations=2)
                f *= good_ones

            xye += [{
                'x': x0,
                'y': y0,
                'h': bb_size[0],
                'w': bb_size[1]
            }]
        crop_frames += [f]
        # crop_frames += [resize(
        #     f,
        #     f_size,
        #     preserve_range=True,
        #     order=0).astype(np.float32)]
    crop_frames, toss_index = drop_empty_frames(crop_frames, list_type=True)
    xye = [d for idx, (d, t) in enumerate(zip(xye, toss_index)) if idx != t]
    return crop_frames, xye, toss_index


def trim_and_threshold(
        data_path,
        skip_frames_beg,
        skip_frames_end,
        low,
        high,
        show_result=False,
        denoise=False,
        remove_objects_smaller_than=400):
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
        im_batch = np.asarray(images[start:start + batch_size])
        if batch_size == 1:
            im_batch = im_batch[None, :, :]
        yield im_batch[:, :, :, None]


def crop_to_shape(img, h0, h1, w0, w1):
    return img[h0:h1, w0:w1]


def mask_to_shape(img, h0, h1, w0, w1):
    mask = np.zeros((img.shape))
    mask[h0:h1, w0:w1] = 1
    return img * mask


def crop_aspect_and_resize_center(img, new_size, resize_or_pad='resize'):
    '''Crop to appropros aspect ratio then resize.'''
    h, w = img.shape[:2]
    ih, iw = new_size
    ideal_aspect = float(iw) / float(ih)
    current_aspect = float(w) / float(h)
    if current_aspect > ideal_aspect:
        new_width = int(ideal_aspect * h)
        offset = (w - new_width) / 2
        crop_h = [0, h]
        crop_w = [offset, w - offset]
    else:
        new_height = int(w / ideal_aspect)
        offset = (h - new_height) / 2
        crop_h = [offset, h - offset]
        crop_w = [0, w]
    cropped_im = img[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
    # return misc.imresize(cropped_im, new_size, mode='F')
    target_dtype = img.dtype
    if resize_or_pad == 'resize':
        return resize(
            cropped_im,
            new_size,
            preserve_range=True,
            order=0).astype(target_dtype)
    elif resize_or_pad == 'pad':
        return pad_image_to_shape(cropped_im, new_size, constant=0.)
    else:
        raise RuntimeError(
            'Cannot understand if you are trying to resize or pad.')


def pad_image_to_shape(cropped_im, new_size, constant=0.):
    cim_size = cropped_im.shape
    pad_top = (new_size[0] - cim_size[0]) // 2
    pad_bottom = (new_size[0] - cim_size[0]) // 2
    pad_left = (new_size[1] - cim_size[1]) // 2
    pad_right = (new_size[1] - cim_size[1]) // 2
    return cv2.copyMakeBorder(
        cropped_im,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, constant)


def process_kinect_tensorflow(model_ckpt, kinect_data, config):
    """Train and evaluate the model."""

    # Import your model
    print 'Model directory: %s' % config.model_output
    print 'Running model: %s' % config.model_type
    model_file = import_cnn(config.model_type)
    if isinstance(kinect_data, basestring):
        use_tfrecords = True
    else:
        use_tfrecords = False
    num_label_targets = config.keep_dims * len(config.joint_order)

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        if use_tfrecords:
            val_data_dict = data_loader_joints.inputs(
                tfrecord_file=kinect_data,
                batch_size=config.validation_batch,
                im_size=config.resize,
                target_size=config.image_target_size,
                model_input_shape=config.resize,
                train=config.data_augmentations,
                label_shape=config.num_classes,
                num_epochs=1,
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
                mask_occluded_joints=config.mask_occluded_joints,
                background_multiplier=config.background_multiplier)
        else:
            val_data_dict = {
                'image': tf.placeholder(
                    dtype=tf.float32,
                    shape=[config.validation_batch] + config.image_target_size[:2] + [1],
                    name='image'),
                'label': tf.constant(np.zeros((1, num_label_targets)))
            }

        # Check output_shape
        if config.selected_joints is not None:
            print 'Targeting joint: %s' % config.selected_joints
            joint_shape = len(config.selected_joints) * config.keep_dims
            if (config.num_classes // config.keep_dims) > (joint_shape):
                print 'New target size: %s' % joint_shape
                config.num_classes = joint_shape
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            print 'Creating training graph:'
            model = model_file.model_struct(
                vgg16_npy_path=config.vgg16_weight_path)
            train_mode = tf.get_variable(name='training', initializer=False)
            model.build(
                rgb=val_data_dict['image'],
                target_variables=val_data_dict,
                train_mode=train_mode,
                batchnorm=[''])
            predictions = model.output

    # Initialize the graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=config.keep_checkpoints)

        # Need to initialize both of these if supplying num_epochs to inputs
        sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
        # saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
        # saver.restore(sess, 'my-save-dir/my-model-10000')

        # Start testing data
        y_batch = []
        yhat_batch = []
        im_batch = []
        num_joints = len(config.joint_order)
        num_batches = len(kinect_data) // config.validation_batch
        normalize_vec = tf_fun.get_normalization_vec(config, num_joints)
        saver.restore(sess, model_ckpt)
        if use_tfrecords:
            # Set up exemplar threading
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            step = 0
            try:
                while not coord.should_stop():
                    it_yhat, it_y, score, it_im = sess.run(
                        [
                            predictions,
                            val_data_dict['label'],
                            tf_fun.l2_loss(
                                predictions, val_data_dict['label']),
                            val_data_dict['image']
                        ])
                    print score
                    if config.normalize_labels:
                        norm_it_yhat = it_yhat * normalize_vec
                        norm_it_y = it_y * normalize_vec
                    yhat_batch += [norm_it_yhat.squeeze()]
                    y_batch += [norm_it_y.squeeze()]
                    im_batch += [it_im]
                    step += 1
            except tf.errors.OutOfRangeError:
                print 'Done with %d steps.' % step
            finally:
                coord.request_stop()
                coord.join(threads)
        else:
            for image_batch in tqdm(
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
                im_batch += [image_batch]
                if config.normalize_labels:
                    norm_it_yhat = it_yhat * normalize_vec
                yhat_batch += [it_yhat]
            y_batch = deepcopy(yhat_batch)

    return {
        'yhat': np.concatenate(yhat_batch).squeeze(),
        'ytrue': np.concatenate(y_batch).squeeze(),
        'im': np.concatenate(im_batch)
        }


def overlay_joints_frames(
        # frames,
        # joint_predictions,
        joint_dict,
        output_folder,
        target_key='yhat'):
    tf_fun.make_dir(output_folder)
    colors, joints, num_joints = monkey_mosaic.get_colors()
    files = []
    frames = joint_dict['im']
    h, w = frames[0].shape
    joint_predictions = joint_dict[target_key]
    for idx, (fr, jp) in tqdm(
            enumerate(zip(frames, joint_predictions)), total=len(frames)):
        f, ax = plt.subplots()
        f.set_dpi(100)
        DPI = f.get_dpi()
        f.set_size_inches(w/float(DPI), h/float(DPI))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.axis('off')
        f.set_tight_layout(True)
        xy_coors = monkey_mosaic.xyz_vector_to_xy(jp)
        if len(fr.shape) > 2:
            im = fr[:, :, 0]
        else:
            im = fr
        ax.imshow(im, cmap='Greys_r')
        [ax.scatter(xy[0], xy[1], color=c) for xy, c in zip(xy_coors, colors)]
        # monkey_mosaic.plot_coordinates(ax, jp, colors)
        out_name = os.path.join(output_folder, '%s.png' % idx)
        plt.savefig(out_name)
        files += [out_name]
        plt.close('all')
    return files


def make_mosaic(images, remove_images=None):
    if remove_images is not None:
        rem_idx = np.ones(len(images))
        rem_idx[remove_images] = 0
        images = images[rem_idx == 1, :, :]

    im_dim = images.shape
    num_cols = np.sqrt(im_dim[-1]).astype(int)
    num_cols = np.min([num_cols, 5])
    num_rows = num_cols
    canvas = np.zeros((im_dim[0] * num_rows, im_dim[1] * num_cols))
    count = 0
    row_anchor = 0
    col_anchor = 0
    for x in range(num_rows):
        for y in range(num_cols):
            it_image = images[:, :, count]
            it_image += (np.sign(np.min(it_image)) * np.min(it_image))
            it_image /= (np.max(it_image) + 1e-3)
            canvas[
                row_anchor:row_anchor + im_dim[0],
                col_anchor:col_anchor + im_dim[1]] += np.sqrt(it_image)
            col_anchor += im_dim[1]
            count += 1
        col_anchor = 0
        row_anchor += im_dim[0]
    return canvas


def plot_filters(layer_weights, title=None, show=False):
    mosaic = make_mosaic(layer_weights)
    plt.imshow(mosaic, interpolation='none', cmap='Greys')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def create_joint_tf_records_for_kinect(
        depth_files,
        depth_file_names,
        model_config,
        kinect_config,
        dummy_label='/media/data_cifs/monkey_tracking/batches/TrueDepth100kStore/labels/pixel_joint_coords/tmp4_000626.npy'):

    """Feature extracts and creates the tfrecords."""
    num_successful = 0
    num_files = len(depth_files)
    tf_file = kinect_config['tfrecord_name']
    max_array = []
    toss_frame_index = []
    label_files = np.asarray([  # replace the depth dir with label dir
        os.path.join(
            model_config.pixel_label_dir,
            re.split(
                model_config.image_extension,
                re.split('/', x)[-1])[0] + model_config.label_extension)
        for x in depth_file_names])
    if not os.path.exists(label_files[0]):
        label_files = np.repeat(dummy_label, len(depth_file_names))
    occlusion_files = np.asarray([
        os.path.join(
            model_config.occlusion_dir,
            re.split(
                model_config.label_extension,
                re.split('/', x)[-1])[0] + model_config.occlusion_extension)
        for x in label_files])
    print 'Saving tfrecords to: %s' % tf_file
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i, depth_image in tqdm(
                enumerate(depth_files),
                total=num_files):

            if depth_image.sum() > 0:
                # extract depth image
                max_array += [depth_image.max()]
                depth_image[depth_image == depth_image.min()] = 0.
                # set nans to 0
                depth_image[np.isnan(depth_image)] = 0.
                if depth_image.shape[:2] != tuple(model_config.image_target_size[:2]):
                    depth_image = resize(
                        depth_image,
                        model_config.image_target_size[:2],
                        preserve_range=True,
                        order=0).astype(np.float32)
                if len(depth_image.shape) < len(model_config.image_target_size):
                    depth_image = np.repeat(
                        depth_image[:, :, None],
                        model_config.image_target_size[-1],
                        axis=-1)
                elif depth_image.shape[-1] < model_config.image_target_size[-1]:
                    num_reps = model_config.image_target_size[-1] // depth_image.shape[-1]
                    depth_image = np.repeat(
                        depth_image,
                        num_reps,
                        axis=-1)[:, :, :num_reps]
                example = data_processing_joints.encode_example(
                    im=depth_image.astype(np.float32),
                    label=np.load(label_files[i]).astype(np.float32),
                    im_label=depth_image.astype(np.float32),
                    occlusion=np.load(occlusion_files[i]).astype(np.float32))
                tfrecord_writer.write(example)
                num_successful += 1
            else:
                toss_frame_index += [i]
    return tf_file, np.max(max_array), np.asarray(toss_frame_index)
