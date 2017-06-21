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


DEPTH_DATA_PATH = '/media/data_cifs/monkey_tracking/extracted_kinect_depth/Xef2Mat_Output_Trial02_np_conversion'


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
    binary_opening(m, output=m)
    remove_small_objects(m, min_size=35, in_place=True)
    binary_closing(m, output=m)
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
        f, (a1, a2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        artists = [[a1.imshow(o), a2.imshow(r)] for o, r in zip(frames, results)]
        ani = animation.ArtistAnimation(f, artists, interval=50)
        plt.show()
        plt.close('all')

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
                 mog_bg_threshold=2.6):
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
    p = Pool()
    results_16bit = p.map(mask_voting, zip(frames_16bit, itertools.repeat(quorum), zip(*mask_lists)))
    p.close()

    # Display animation if requested
    if show_result:
        print('Displaying result...')
        plt.close('all')
        fig, (a1, a2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
        artists = [[a1.imshow(f), a2.imshow(r)]
                   for f, r, in zip(frames_16bit, results_16bit)]
        a1.set_title('Orig'); a2.set_title('Res')
        ani = animation.ArtistAnimation(fig, artists, interval=50)
        plt.show()
        plt.close('all')

    return results_16bit


def best_box(frame, w, h, _x, _y, x_, y_):
    best = 0
    X, Y = 0, 0
    for y in range(_y, y_ - h):
        if frame[_x-w:_x,y:y+h].sum() > best:
            best = frame[_x-w:_x,y:y+h].sum()
            X, Y = _x - w, y
    for y in range(_y, y_ - h):
        if frame[x_:x_+w,y:y+h].sum() > best:
            best = frame[x_:x_+w,y:y+h].sum()
            X, Y = _x, y
    for x in range(_x, x_ - w):
        if frame[x:x+w,_y-h:_y].sum() > best:
            best = frame[x:x+w,_y-h:_y].sum()
            X, Y = x, _y - h
    for y in range(_x, x_ - w):
        if frame[x:x+w,y_:y_+h].sum() > best:
            best = frame[x:x+w,y_:y_+h].sum()
            X, Y = x, y_
    return frame[X:X+w, Y:Y+h]


def box_tracking(frames, w, h, _x, _y, x_, y_):
    '''
    For each frame, find the `w` by `h` box with the largest sum,
    out of all boxes in the region consisting of the rectangles
    described by the following pairs of opposite corners:
        (_x - w, _y), (_x, y_)
        (x_, _y), (x_ + w, y_)
        (_x, _y - h), (x_, _y)
        (_x, y_), (x_, y_ + h)
    '''
    return [best_box(f, w, h, _x, _y, x_, y_) for f in frames]


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


if __name__ == '__main__':
    frames = get_and_trim_frames(DEPTH_DATA_PATH, 100, 35)
    # bg = static_background(frames)
    # print(bg.shape)
    # plt.figure(); plt.imshow(bg); plt.show(); plt.close()
    # frames = [bg] + frames
    threshed = threshold(frames, 1400, 3500, show_result=False, denoise=True)
    plt.close('all')
    # res = bgsub_frames(threshed, 1, 1, True, 0.1)
    cropped = box_tracking(frames, 100, 100, 120, 50, 300, 400)
    f, a = plt.subplots(1, 1)
    artists = [[plt.imshow(c)] for c in cropped]
    ani = animation.ArtistAnimation(f, artists, interval=50)
    plt.show()
