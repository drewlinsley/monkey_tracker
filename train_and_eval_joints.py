import argparse
from ops.tf_model_cnn_joints import train_and_eval
from ops.data_processing_joints import process_data
from config import monkeyConfig


def main(extract_features=False):
    config = monkeyConfig()

    # Encodes files into tfrecords
    if extract_features == True:
        print '-'*60
        print 'Creating new tfrecords file'
        print '-'*60
        process_data(config)

    # Trains a random forest model
    train_and_eval(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract",
        dest="extract_features",
        action='store_true',
        help='Extract features -> tfrecords or reuse existing.')
    args = parser.parse_args()
    main(**vars(args))
