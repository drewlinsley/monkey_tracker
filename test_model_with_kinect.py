import argparse
from ops.tf_model_cnn_joints import train_and_eval
from config import monkeyConfig


def main(model_dir):
    '''Skeleton script for preprocessing and
    passing kinect videos through a trained model'''
    config = monkeyConfig()

    # Encodes files into tfrecords
    train_and_eval(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        help='Name of model directory.')
    args = parser.parse_args()
    main(**vars(args))
