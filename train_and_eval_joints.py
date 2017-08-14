import argparse
from ops.tf_model_cnn_joints import train_and_eval
from ops.data_processing_joints import process_data
from config import monkeyConfig


def main(extract_features=False, which_joint=None, babas_data=False):
    config = monkeyConfig()

    # Encodes files into tfrecords
    if extract_features == True:
        print '-'*60
        print 'Creating new tfrecords file'
        print '-'*60
        process_data(config)

    if which_joint is not None:
        config.selected_joints += [which_joint]
    train_and_eval(config, babas_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract",
        dest="extract_features",
        action='store_true',
        help='Extract features -> tfrecords or reuse existing.')
    parser.add_argument(
        "--babas",
        dest="babas_data",
        action='store_true',
        help='Train on special babas data (depreciated).')
    parser.add_argument(
        "--which_joint",
        dest="which_joint",
        type=str,
        help='Specify a joint to target with the model.')

    args = parser.parse_args()
    main(**vars(args))
