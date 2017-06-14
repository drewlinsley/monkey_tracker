import argparse
from ops.tf_model_cnn_joints import train_and_eval
from config import monkeyConfig
import cv2
import os


def main(model_dir):
    '''Skeleton script for preprocessing and
    passing kinect videos through a trained model'''
    config = monkeyConfig()
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
        help='Name of model directory.')
    args = parser.parse_args()
    main(**vars(args))
