"""
Utility for converting Kinect data to python numpys.
"""

import argparse
import os
import numpy as np
import scipy.io as sio
from glob import glob
from ops.tf_fun import make_dir
from tqdm import tqdm


def main(file_dir, wildcard):
    """
    Execute kinect-file conversion.

    Example usage:
    python ops/kinect_util_scripts/convert_mat_to_npy.py --file_dir=/media/data_cifs/monkey_tracking/extracted_kinect_depth/201710091108-Freely_Moving_Recording_depth_0 --wildcard=DepthFrame
    """
    files = glob(os.path.join(file_dir, '%s*.mat' % wildcard))
    out_dir = os.path.join(file_dir, '%s_npys' % wildcard)
    make_dir(out_dir)
    print 'Saving .npys to %s' % out_dir
    for f in tqdm(files, total=len(files)):
        mat = sio.loadmat(f)
        npy_name = f.strip('.mat').split('/')[-1]
        out_name = os.path.join(out_dir, npy_name)
        vals = [v for v in mat.values() if isinstance(v, np.ndarray)]
        np.save(out_name, vals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_dir",
        dest="file_dir",
        type=str,
        default=None,
        help='Path to directory with .mat files.')
    parser.add_argument(
        "--wildcard",
        dest="wildcard",
        type=str,
        default=None,
        help='Wildcard for searching for files.')
    args = parser.parse_args()
    main(**vars(args))
