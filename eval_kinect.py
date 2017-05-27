import argparse
from ops.eval_tf_model_fc import eval_model
from config import monkeyConfig


def main(extract_features, show_figures):
    config = monkeyConfig()
    eval_model(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--show_figures', dest='show_figures',
        default=None, help='Force produce figures.')
    args = parser.parse_args()
    main(**vars(args))
