import argparse
from ops.data_processing_joints import process_data
from config import monkeyConfig


def main(extract_features=False, which_joint=None, hp_optim=False):
    config = monkeyConfig()

    # Encodes files into tfrecords
    if extract_features == True:
        print '-'*60
        print 'Creating new tfrecords file'
        print '-'*60
        process_data(config)

    if hp_optim:
        from ops.tf_model_cnn_joints_multi_gpu_hp_optim import train_and_eval
    else:
        from ops.tf_model_cnn_joints_multi_gpu import train_and_eval

    if which_joint is not None:
        config.selected_joints += [which_joint]
    train_and_eval(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hp_optim",
        dest="hp_optim",
        action='store_true',
        help='Trigger a hp_optim worker.')
    parser.add_argument(
        "--extract",
        dest="extract_features",
        action='store_true',
        help='Extract features -> tfrecords or reuse existing.')
    parser.add_argument(
        "--which_joint",
        dest="which_joint",
        type=str,
        help='Specify a joint to target with the model.')
    args = parser.parse_args()
    main(**vars(args))
