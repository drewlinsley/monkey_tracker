## This script creates (approximate) occlusion maps, converts maya coordinates to (approximate) pixel coordinates 
## and converts depth image pixel intensities to actual depth coordinates

import numpy as np 
import os
import argparse
from tqdm import tqdm
from multiprocessing import Process, Value, Lock, Pool
from multiprocessing.pool import ThreadPool
import threading
from tifffile import TiffFile
from scipy.spatial.distance import cdist
import scipy.misc as misc
from scipy import stats
from PIL import Image
import sys, inspect
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0,parentdir) 
from config import monkeyConfig

config = monkeyConfig()
joints = config.joint_order
keys = np.asarray(config.labels.keys())
key_map = {}
values = []
for i, key in enumerate(keys):
	key_map[key] = i
	values.append(config.labels[key])
# this maps a joint name to an array of numbers. The numbers represent indices of
# elements the values array that the joint can be labeled as and not be considered occluded
occlusion_map = {
    'lEye' : [key_map['head']], 
    'neck' : [key_map['neck']], 
    'abdomen' : [key_map['hip']], 
    'lShldr' : [key_map['left_shoulder'], key_map['left_upper_arm']], 
    'rShldr' : [key_map['right_shoulder'], key_map['right_upper_arm']], 
    'lForeArm' : [key_map['left_upper_arm'], key_map['left_lower_arm']], 
    'rForeArm' : [key_map['right_upper_arm'], key_map['right_lower_arm']], 
    'lHand' : [key_map['left_lower_arm'], key_map['left_hand']], 
    'rHand' : [key_map['right_lower_arm'], key_map['right_hand']], 
    'lMid1' : [key_map['left_hand'], key_map['left_finger']], 
    'rMid1' : [key_map['right_hand'], key_map['right_finger']],
    #I'm letting the fingers be labeled as either hand or finger - just because they're so small
    # and might often get labeled as one or the other 
    'lMid3' : [key_map['left_hand'], key_map['left_finger']], 
    'rMid3' : [key_map['right_hand'], key_map['right_finger']], 
    'lThigh' : [key_map['hip'], key_map['left_upper_leg']], 
    'rThigh' : [key_map['hip'], key_map['right_upper_leg']],
    'lShin' : [key_map['left_upper_leg'], key_map['left_lower_leg']], 
    'rShin' : [key_map['right_upper_leg'], key_map['right_lower_leg']], 
    'lFoot' : [key_map['left_lower_leg'], key_map['left_foot']], 
    'rFoot' : [key_map['right_lower_leg'], key_map['right_foot']], 
    'lToe' : [key_map['left_foot'], key_map['left_toe']], 
    'rToe' : [key_map['right_foot'], key_map['right_toe']], 
    'lToeMid3' : [key_map['left_toe']], 
    'rToeMid3' : [key_map['right_toe']]
}

labelImDir = []
coordDir = []
pixelDir = []
occlusionDir = []
npyLabelDir = []
depthImDir = []
adjDepthDir = []

im_ext = '.tif'

# class counter():
# 	def __init__(self):
# 		self.val = Value('i', 0)
# 		self.lock = Lock()

def process(filename):
	coordFile = os.path.join(coordDir[0], filename)
	labelImFile = os.path.join(labelImDir[0], filename[:-4]+im_ext)
	depthImFile = os.path.join(depthImDir[0], filename[:-4]+im_ext)
	coords = np.load(coordFile)
	image = TiffFile(labelImFile).asarray() / 255
	depth = TiffFile(depthImFile).asarray().astype(np.float32)

	pixel_coords = (coords[:, :2]*np.asarray([-622.34446575, -622.36193241])/coords[:, 2].reshape([23, 1]) + 
		np.asarray([319.50719891, 239.50096976]))
	pixel_coords = np.concatenate([pixel_coords, coords[:, 2].reshape([23, 1])], axis=1).astype(np.int32)

	# select all pixel coordinates that are in the bounds of the image
	in_bound_bool = (pixel_coords[:, 0] < 640) * (pixel_coords[:, 0] > -1) * (pixel_coords[:, 1] < 480) * (pixel_coords[:, 1] > -1)
	# labels is the value of the image at each pixel coordinate, when that coordinate is in range. It
	# treats any out-of-range pixel coordinates as background
	# the image is loaded as its transpose for some reason
	labels = np.asarray([image[x[1], x[0]] if in_bounds else [0, 0, 0, 0] for (x, in_bounds) in zip(pixel_coords, in_bound_bool)])
	labels = np.argmin(cdist(labels, values, 'euclidean'), axis=1)
	visible = [label in occlusion_map[joints[i]] for i, label in enumerate(labels)]
	
	# These were set in the maya animation
	nClip = 1.0; fClip = 1300.0;
	# take out any noise values in the image
	monkeyVal, _ = stats.mode(depth[depth[:, :, 3] != 0][:, 3], axis=None)
	depth[np.logical_or(depth[:, :, 3] > monkeyVal+1., depth[:, :, 3] < monkeyVal-1.)] = [0., 0., 0., 0.]
	# convert pixel intensity to depth
	depth[depth[:, :, 3] != 0] = depth[depth[:, :, 3] != 0] * ((nClip - fClip) / 2.0**16) + fClip

	np.save(os.path.join(pixelDir[0], filename), pixel_coords) 
	np.save(os.path.join(occlusionDir[0], filename), visible) 
	np.save(os.path.join(npyLabelDir[0], filename), image)
	np.save(os.path.join(adjDepthDir[0], filename), depth)
	# global active_threads
	# with active_threads.lock:
	# 	active_threads.val.value -= 1
	# os.exit(0)

def main(parentDir):
	# global active_threads
	# active_threads = counter()
	labelDir = os.path.join(parentDir, 'labels')
	depthDir = os.path.join(parentDir, 'depth')
	labelImDir.append(os.path.join(labelDir, 'layer2'))
	coordDir.append(os.path.join(labelDir, 'joint_coords'))
	pixelDir.append(os.path.join(labelDir, 'pixel_joint_coords'))
	occlusionDir.append(os.path.join(labelDir, 'occlusions'))
	npyLabelDir.append(os.path.join(labelDir, 'npyLabels'))
	depthImDir.append(os.path.join(depthDir, 'layer1'))
	adjDepthDir.append(os.path.join(depthDir, 'true_depth'))

	os.system("mkdir -p %s" % pixelDir[0]) 
	os.system("mkdir -p %s" % occlusionDir[0])
	os.system("mkdir -p %s" % npyLabelDir[0])
	os.system("mkdir -p %s" % adjDepthDir[0])
	# go through each 3D coordinate file and save its pixel coordinates and whether
	# each joint is visible
	p = Pool(10)
	files = os.listdir(coordDir[0])
	for _ in tqdm(p.imap_unordered(process, files), total=len(files)):
		pass
	# for filename in tqdm(files):
	# 	t = Process(target=process, args=(filename,))
	# 	while active_threads >= 100:
	# 		pass
	# 	with active_threads.lock:
	# 		active_threads.val.value += 1
	# 	t.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('parentDir',type=str,
        help='Directory containing all the depth and label renders')
    args = parser.parse_args()
    main(**vars(args))
