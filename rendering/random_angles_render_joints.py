#######PLEASE READ ALL OF THIS BEFORE USING##########


######need to run this using the mayapy application, which is essentially an alternate version
## of python, located at /usr/autodesk/maya2017/bin/mayapy
## so instead of typing 'python random_angles_render.py' you would type '/usr/autodesk/maya2017/bin/mayapy random_angles_render_joints.py'
## or, since I added it to the bin, you could just do 'mayapy random_angles_render.py'


##this script also assumes that the user has already created an animation layer and set it up for depth rendering
## to do this, go to the layer editor tab on the right. Then click the render tab on the bottom. Create a new render layer,
## and add the animated objects. Then right click it and click attributes. Select presets, and then click luminance depth.
## (you'll want to make sure the monkey mesh is included in the layer).
## Make sure you're using this new layer (the monkey will appear black), and click Window->rendering editors->render settings
## set frame/animation set to "name_#.ext", and set frame padding to 3 or more. Then close, and you're all set

## the animation should also have its depth render layer called layer1, and it should have another render layer called layer2 which 
## is normal, but which has no shading.

## The width of the window that the render is being taken of should be 500 Maya units and 
## front orthographic view in our experiments

## execute this script using sudo


import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages/')
import os
import argparse
import numpy as np
import maya.standalone
import maya.cmds as cmds
import random as rand

def main(infile, numFrames, prefix, folderName):
	#have to initialize a standalone session in order for cmds to work
	maya.standalone.initialize()

	cmds.file(infile, o=True)

	## orient the monkey to start (see assumptions at top)
	anchor = cmds.ls('SKMonkey_31830')[0]

	cmds.currentTime(1, edit=True)
	currentTime = 1
	#rotate the anchor. a=True means the rotation is absolute (not relative to current position), and 
	#ws=True means rotate based on the global axis, rather than the axis of the joint itself
	# cmds.rotate(0,0,0, anchor, a=True, ws=True) #note1

	##from now on we will rotate the monkey from its hips and assume that the anchor won't be rotated again
	hip = cmds.ls('hip')[0] 
	#look at the monkey from an upper angle. we will want to rotate it at limited angles about the x axis, and 
	# at every angle about the y axis
	# x now represents pitch, and y represents horizontal viewing angle
	cmds.rotate(-15, 0,0, hip, r=True, ws=True) 

	numPitches = 5
	numAngles = 10 
	numZooms = 10
	numShifts = 10

	## orient the monkey to start (see assumptions at top)
	labels = ['bregma', 'scapula', 'iliac crest', 'left humerus head', 'right humerus head', 'left elbow', 'right elbow', 'left distal ulna head', 
	'right distal ulna head', 'left metacarpophalangeal', 'right metacarpophalangeal', 'left digit tip', 'right digit tip', 'left trochanter major',
	'right trochanter major', 'left knee', 'right knee', 'left malleolus', 'right malleolus', 'left metatarsal', 'right metatarsal', 'left toe tip', 
	'right toe tip']

	jointNames = ['lEye', 'neck', 'abdomen', 'lShldr', 'rShldr', 'lForeArm', 'rForeArm', 'lHand', 'rHand', 'lMid1', 'rMid1', 'lMid3', 'rMid3', 'lThigh', 'rThigh',
	'lShin', 'rShin', 'lFoot', 'rFoot', 'lToe', 'rToe', 'lToeMid3', 'rToeMid3']

	def update(coordArray):
		# treating the left and right eye specially to average them
		for i in range(len(jointNames)):
			coordArray[i] = np.array(cmds.joint(jointNames[i], q=True, p=True, a=True))

	outdir = folderName #infile.split("/")[-1].split(".")[0]

	#outdir = outdir + '-big'
	pathPrefix = '/media/data_cifs/monkey_tracking/batches/'
	os.system("mkdir -p "+pathPrefix + outdir) 
	os.system("mkdir -p "+pathPrefix + outdir + '/labels/joint_coords')
	print("made dir")

	# select 5 random pitches, sampling from 5 ranges: from 10-12, 12-14, 14-16, 16-18, and 18-20 degrees
	# for each pitch, sample 36 random rotations from 36 ranges: from 0-10, 10-20, 20-30, etc.
	# total will be 180 renderings, each one containing as many images as there were frames in the animation
	# to save time, we copy the animation, and insert it at a later time, then rotate the monkey for that
	# duplicate animation; this way we don't have to keep opening the file
	for i in range(1, numPitches*numAngles*numZooms*numShifts): 
		cmds.copyKey(anchor, hi='below', t=(0, numFrames)) ## we may not have to copy repeatedly, take it out of loop?
		cmds.pasteKey(anchor, option='insert', time=(numFrames*i+1, numFrames*i+1))
	for i in range(numPitches):   
		rpitch = -1.0*(rand.uniform(0.0, 10.0 / numPitches) + i*(10.0 / numPitches) + 10.0)
		for j in range(numAngles):
			rrotation = rand.uniform(0.0, 360.0 / numAngles) + j*(360.0 / numAngles)
			for z in range(numZooms):					
				## ideally, we would zoom anywhere between 50 and 450, but this cuts off the monkey for being too far away,
				## so we zoom from just 0 to 25 
				rzoom = 300 + rand.uniform(0.0, 700.0 / numZooms)*z 
				for s in range(numShifts): 
					# I calculated that the view expands out on the x axis with a slope of .5 as you go back, with a threshhold of 2.3 at depth 5
					# also the view expands on the y axis with a slope of .35 as you go back, with a threshhold of 1.75 at depth 5
					# we want it to be well within these ranges
					xlim = 0.5*rzoom - 50
					ylim = 0.35*rzoom - 50
					xshift = rand.uniform(-xlim, xlim) 
					yshift = rand.uniform(-ylim, ylim)
					print("pitch "+str(i)+", angle "+str(j)+", zoom "+str(z)+", shift "+str(s))
					for k in range(numFrames):
						cmds.rotate(rpitch, rrotation, 0, hip, a=True, ws=True) 
						cmds.move(rzoom, hip, z=True, a=True, ws=True)
						cmds.move(xshift, hip, x=True, a=True, ws=True)
						cmds.move(yshift, hip, y=True, a=True, ws=True)
						cmds.setKeyframe('hip', ott="step") 
						coordArray = np.zeros((len(labels), 3))
						update(coordArray)
						np.save(pathPrefix+outdir+'/labels/joint_coords/tmp' + str(prefix) + '_'+str(currentTime).zfill(6), coordArray)
						currentTime += 1
						cmds.currentTime(currentTime, edit=True)

	# save the edits to a temporary file that we can run the renderer on
	newName = pathPrefix+outdir+"/tmp"+str(prefix)+".ma"
 
	##rendering for color labels
	cmds.file(rename=newName)
	cmds.file(save=True, type='mayaAscii')

	## again using the software renderer because the colors are different, and the hardware renderer either doesn't render cerain colors or it show
	## a symbol representing a light that shouldn't be in there
	##this step is fast
	os.system("/usr/autodesk/maya2017/bin/Render -r sw -rd %s -rl layer2 -fnc name_#.ext -of png -s 1 -e %s -b 1 -pad 6 %s &" % 
		(os.path.join(pathPrefix, outdir, 'labels'), str(numFrames*numPitches*numAngles*numZooms*numShifts), newName))

	##rendering for depth maps (doesn't seem to work with hardware)
	##this step is slow
	os.system("/usr/autodesk/maya2017/bin/Render -r sw -rd %s -rl layer1 -fnc name_#.ext -of png -s 1 -e %s -b 1 -pad 6 %s &" % 
		(os.path.join(pathPrefix, outdir, 'depth'), str(numFrames*numPitches*numAngles*numZooms*numShifts), newName))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile',type=str,
        help='Directory of file to render')
    parser.add_argument('numFrames',type=int,
        help='Number of frames in animation')
    parser.add_argument('prefix',type=int,
        help='Prefix number to put in front of render names')
    parser.add_argument('folderName',type=str,
        help='Name of folder in which to put renderings')
    args = parser.parse_args()
    main(**vars(args))


