from __future__ import print_function
import numpy as np
import cv2, os, operator, itertools
from multiprocessing import Pool
from skimage.restoration import denoise_tv_bregman
from skimage.morphology import remove_small_objects
from scipy.ndimage.morphology import binary_opening, binary_closing, binary_fill_holes
from glob import glob
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn


def denoise_mask(mask):
    return binary_opening(mask.astype(np.bool).astype(np.int))


def mask_voting(tup):
    '''mask_voting
    Helper for `bgsub_frames(...)`. Takes care of denoising
    masks produced by MOG and voting.
    `tup` should contain a frame, a numeric quorum, and a
    tuple of masks.
    '''
    frame, quorum, morpho_steps, masks = tup
    r = np.zeros_like(frame)
    m = np.zeros_like(masks[0])
    for mm in masks:
        m[mm.astype(np.bool)] += 1
    # voting
    m = m >= quorum
    binary_opening(m, output=m, iterations=morpho_steps)
    remove_small_objects(m, min_size=35, in_place=True)
    binary_closing(m, output=m, iterations=morpho_steps)
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
    if in_image.dtype == np.uint8: return in_image # just in case
    mn, mx = in_image.min(), in_image.max()
    in_image -= mn
    return np.floor_divide(in_image, (mx - mn + 1.) / 256., casting='unsafe').astype(np.uint8)


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
    files = (f for i, f in enumerate(files) if (i >= skip_frames_beg and i < n - skip_frames_end))
    # load up frames
    return [np.load(f) for f in files]


def threshold(frames, low, high,
              show_result=False, denoise=False,
              remove_objects_smaller_than=48):
    '''threshold
    Zero out values which are greater than `high` or smaller than `low`
    in each frame in `frame`. Do denoising per `denoise` and `remove_objects_smaller_than`.
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
            good_ones = binary_closing(binary_opening(binary), iterations=2).astype(np.bool)
        if remove_objects_smaller_than > 0:
            good_ones = remove_small_objects(good_ones, min_size=remove_objects_smaller_than)
        results[i][good_ones] = f[good_ones]

    if show_result:
        print('Displaying result...')
        multianimate([frames, results], ['Original', 'Results'])

    return results

def trim_and_threshold(data_path, skip_frames_beg, skip_frames_end, low, high,
                       show_result=False, denoise=False,
                       remove_objects_smaller_than=48):
    '''trim_and_threshold
    Get data and remove start and end frames with `get_and_trim_frames`,
    and then pass all that to `threshold`.
    '''
    frames = get_and_trim_frames(data_path, skip_frames_beg, skip_frames_end)
    return threshold(frames, low, high, show_result, denoise, remove_objects_smaller_than)


def bgsub_frames(frames_16bit, wraps=32, quorum=10, show_result=False,
                 mog_bg_threshold=2.6, morpho_steps=1):
    '''bgsub_frames
    Given a list of nparrays `frames_16bit`, create `wraps` MOG background
    subtractors with selectivity `mog_bg_threshold` (documented in cv2 docs),
    and have them vote so that a pixel remains in the output if `quorum` of them
    agree that it should.
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
        # fgbg = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=mog_bg_threshold, detectShadows=False)
        offset_frames = frames_8bit[offset:] + frames_8bit[:offset]
        offset_masks = [fgbg.apply(frame) for frame in offset_frames]
        mask_lists.append(offset_masks[-offset:] + offset_masks[:-offset])

    # and, or those masks to mask 16 bit originals
    print('Masking...')
    results_16bit = map(mask_voting,
                        zip(frames_16bit, 
                            itertools.repeat(quorum),
                            itertools.repeat(morpho_steps),
                            zip(*mask_lists)))

    # Display animation if requested
    if show_result:
        print('Displaying result...')
        multianimate([frames_16bit, results_16bit], ['Original', 'Result'])

    return results_16bit


def best_box(frame, w, h, _x, _y, x_, y_, binary=None, ignore_border_px=0):
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
        if binary[x_- w:x_, y:y + hh].sum() > best:
            best = binary[x_- w:x_, y:y + hh].sum()
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


def box_tracking(frames, w, h, _x, _y, x_, y_,
                 binaries=None, ignore_border_px=0):
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
        return [best_box(f, w, h, _x, _y, x_, y_, ignore_border_px=ignore_border_px)
                for f in frames]
    else:
        return [best_box(f, w, h, _x, _y, x_, y_, b, ignore_border_px)
                for b, f in zip(binaries, frames)]


def trim_and_bgsub(data_path, skip_frames_beg, skip_frames_end, wraps,
                   show_result=False, denoise=True):
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
    frames_16bit = get_and_trim_frames(data_path, skip_frames_beg, skip_frames_end)
    return bgsub_frames(frames_16bit, denoise, show_result, wraps)


def multianimate(alolom, titles=None, figtitle=None, show=True, save_to=None,
                 repeat_if_show=True, **figure_kwargs):
    '''multianimate
    Helper for showing simultaneous videos on different axes with imshow,
    nice for comparing results.
    `alolom` is a list of lists of matrices to be displayed.
    `titles` is a list of titles corresponding to each list of matrices.
    '''
    n = len(alolom)
    if titles is None:
        titles = range(1, n + 1)
    
    # arrangement
    if n % 3 is 0:
        c, r = 3, n / 3
    elif n % 2 is 0:
        c, r = 2, n / 2
    else:
        c, r = 1, n
    
    # with squeeze=False, we have same dims
    alololom, titles = zip(*([iter(alolom)] * c)), zip(*([iter(titles)] * c))
    fig, axes = plt.subplots(r, c, squeeze=False,
                             **figure_kwargs)
    # fig.tight_layout()
    if figtitle: fig.suptitle(figtitle)

    # create artists
    artists = []
    for aloa, alot, alolom in zip(axes, titles, alololom):
        for a, t, alom in zip(aloa, alot, alolom):
            a.set_title(t)
            a.set_aspect('equal')
            artists.append([a.imshow(m) for m in alom])

    # animate
    ani = animation.ArtistAnimation(fig, zip(*artists), interval=50, repeat=repeat_if_show)
    if show: plt.show()
    if save_to: ani.save(save_to, 'ffmpeg', 24)
    plt.close(fig)
