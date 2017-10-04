** Add the folders in monkey_tracker/ to your python path **
Add this line (for example) to your .bashrc file : export PYTHONPATH="${PYTHONPATH}:/home/xxx/monkey_tracker/ops"

For testing a new incoming video:
-> Convert the *.mats to *.npys
	refer to ops/kinect_util_scripts/convert_mat_to_npy.py for sample usage
-> encode into a tfrecord


Modifications
*****
from ops.kinect_util_scripts import joint_list
*****
