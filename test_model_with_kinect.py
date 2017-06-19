import argparse
import os
import re
from config import monkeyConfig
from glob import glob
from ops import kinect_video_processing



def main(model_dir, ckpt_name):
    '''Skeleton script for preprocessing and
    passing kinect videos through a trained model'''
    config = monkeyConfig()
    monkey_files = glob(
        os.path.join(
            config.kinect_directory,
            config.kinect_project,
            '*%s' % config.kinect_file_ext))
    monkey_files = sorted(
        monkey_files, key=lambda name: int(
            re.search('\d+', name.split('/')[-1]).group()))
    import ipdb;ipdb.set_trace()

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(
        os.path.join(config.kinect_directory,config.kinect_project,config.kinect_video))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        from matplotlib import pyplot as plt
        import ipdb;ipdb.set_trace()
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default='/media/data_cifs/monkey_tracking/results/TrueDepth100kStore/model_output/cnn_multiscale_high_res_low_res_skinny_pose_occlusion_2017_06_18_17_45_17',
        help='Name of model directory.')
    parser.add_argument(
        "--ckpt_name",
        dest="ckpt_name",
        type=str,
        default='model_56000.ckpt-56000',
        help='Name of TF checkpoint file.')
    args = parser.parse_args()
    main(**vars(args))
