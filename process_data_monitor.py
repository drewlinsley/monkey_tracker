import os
import argparse
from ops.data_processing_joints import process_data_monitor
from config import monkeyConfig
from multiprocessing import Process
import time

def monitor(directory, total_files):
	monitor_file = open(os.path.join(directory, 'monitor.txt'), 'w+')
	depth_directory = os.path.join(directory, 'depth', 'true_depth')
	found_files = {}
	while len(found_files) < total_files:
		for filename in os.listdir(depth_directory):
			if filename not in found_files:
				monitor_file.write(os.path.join(depth_directory, filename)+'\n')
				found_files[filename] = True
		time.sleep(60)

def main(directory, total_files):
	config = monkeyConfig()
	p1 = Process(target=monitor, args=(directory, total_files))
	p1.start()
	p2 = Process(target=process_data_monitor, args=(config, os.path.join(directory, 'monitor.txt'), total_files))
	p2.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
       	type=str,
        help='directory to monitor')
    parser.add_argument(
        "total_files",
        type=int,
        help='total number of files to wait for')
    args = parser.parse_args()
    main(**vars(args))