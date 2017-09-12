"""
Utility function for splitting long kinect movies into multiple files.
"""

import os
import shutil
import numpy as np
from glob import glob
from tqdm import tqdm


def make_dir(x):
    if not os.path.exists(x):
        os.makedirs(x)


def copy_files(source, dest):
    for s, d in zip(source, dest):
        shutil.copy2(s, d)


head_dir = '/media/data_cifs/monkey_tracking/extracted_kinect_depth'
folder = 'starbuck_pole_new_configuration_competition'

depth_key = 'DepthFrame'
IR_key = 'IRFrame'
ext = '.mat'
splits = 10

depth_files = np.asarray(
    glob(os.path.join(head_dir, folder, '%s*%s' % (depth_key, ext))))
IR_files = np.asarray(
    glob(os.path.join(head_dir, folder, '%s*%s' % (IR_key, ext))))

assert len(depth_files) == len(IR_files), 'Different # of depth/IR frames.'
depth_idx = np.argsort(
    [int(x.split(depth_key)[-1].strip(ext)) for x in depth_files])
depth_files = depth_files[depth_idx]
IR_files = IR_files[depth_idx]
num_splits = np.ceil(len(depth_files) / splits)
split_vec = np.arange(splits).repeat(num_splits)
depth_files = depth_files[:len(split_vec)]
IR_files = IR_files[:len(split_vec)]

for idx in tqdm(range(splits), desc='Copying splits'):
    it_depth = depth_files[split_vec == idx]
    it_IR = IR_files[split_vec == idx]
    it_depth_dir = os.path.join(head_dir, '%s_depth_%s' % (folder, idx))
    it_IR_dir = os.path.join(head_dir, '%s_IR_%s' % (folder, idx))
    make_dir(it_depth_dir)
    make_dir(it_IR_dir)
    move_it_depth = [os.path.join(
        it_depth_dir, x.split('/')[-1]) for x in it_depth]
    move_it_IR = [os.path.join(
        it_IR_dir, x.split('/')[-1]) for x in it_IR]
    copy_files(source=it_depth, dest=move_it_depth)
    copy_files(source=it_IR, dest=move_it_IR)
    break
