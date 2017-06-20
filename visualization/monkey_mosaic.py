import os
import re
import argparse
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from config import monkeyConfig
import matplotlib.cm as cm


def save_mosaic(
        ims,
        yhats,
        ys,
        output,
        wspace=0.,
        hspace=0.):
    # Get a color for each yhat and a color for each ytrue
    joints = monkeyConfig().joint_names
    num_joints = len(joints)
    colors = cm.rainbow(np.linspace(0, 1, num_joints))
    rc = np.ceil(np.sqrt(len(ims))).astype(int)
    fig = plt.figure(figsize=(10, 10))
    gs1 = gridspec.GridSpec(rc, rc)
    for idx, (im, yhat, y) in enumerate(zip(ims, yhats, ys)):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(np.log10(im), cmap='Greys_r')
        lab_legend_artists = plot_coordinates(
            ax1, y, colors, marker='.', markersize=1.5)
        est_legend_artists = plot_coordinates(
            ax1, yhat, colors,
            linestyle='none',
            markeredgewidth=.5,
            marker='o',
            mfc='none',
            markersize=1.5)
    plt.subplots_adjust(top=1, left=0)

    # Legend
    patches = est_legend_artists + lab_legend_artists
    fig.legend(
        patches,
        (["" for _ in colors] +
         [j for j in joints]),
        title="Estimated / True",
        loc=1, frameon=False, numpoints=1, ncol=2,
        columnspacing=0, handlelength=0.25, markerscale=2)
    plt.savefig(output)


def xyz_vector_to_xy(vector, num_dims=2):
    return vector.reshape(-1, num_dims)[:, :2]


def plot_coordinates(ax, vector, colors, **kwargs):
    xy = xyz_vector_to_xy(vector)
    # it will be helpful to have this list of paths
    # for legend making
    return [ax.plot(
                x,
                y,
                color=c,
                mec=c,
                **kwargs)[0]
            for c, (x, y) in zip(colors, xy)]


def main(
        monkey_date,
        num_files=1,
        output_file='monkey_mosaic.png',
        dmurphy_npy_dir='/media/data_cifs/monkey_tracking/batches/test',
        normalize=False,
        unnormalize=False,
        max_ims=4
        ):

    monkey_dir = os.path.join(dmurphy_npy_dir, monkey_date)
    # Eventually read settings from the saved config file
    if unnormalize:
        # Because normalization was fucked up in the training script
        config = monkeyConfig()
        unnormalize_vec = np.asarray(
            config.image_target_size[:2] + [config.max_depth]).repeat(
            len(config.joint_order))
        normalize_vec = np.asarray(
            config.image_target_size[:2] + [config.max_depth]).reshape(
            1, -1).repeat(23, axis=0).reshape(1, -1)
    if normalize:
        config = monkeyConfig()
        normalize_vec = np.asarray(
            config.image_target_size[:2] + [config.max_depth]).reshape(
            1, -1).repeat(23, axis=0).reshape(1, -1)
    ims = glob(os.path.join(monkey_dir, 'im_*'))
    ims.sort(key=os.path.getmtime)
    ims = ims[::-1][:num_files]
    im_list = []
    yhat_list = []
    ytrue_list = []
    for im in ims:
        file_name = re.search('\d+', im.split('/')[-1]).group()
        images = np.squeeze(np.load(im))
        yhats = np.load(os.path.join(monkey_dir, 'yhat_%s.npy' % file_name))
        ytrues = np.load(os.path.join(monkey_dir, 'ytrue_%s.npy' % file_name))
        if normalize:
            yhats *= normalize_vec
            ytrues *= normalize_vec
        if unnormalize:
            yhats /= unnormalize_vec
            ytrues /= unnormalize_vec
            yhats *= normalize_vec
            ytrues *= normalize_vec
        [im_list.append(x) for x in images]
        [yhat_list.append(x) for x in yhats]
        [ytrue_list.append(x) for x in ytrues]

    if max_ims is not None:
        rand_order = np.random.permutation(len(im_list))
        im_list = np.asarray(im_list)[rand_order][:max_ims]
        yhat_list = np.asarray(yhat_list)[rand_order][:max_ims]
        ytrue_list = np.asarray(ytrue_list)[rand_order][:max_ims]

    save_mosaic(
        ims=im_list,
        yhats=yhat_list,
        ys=ytrue_list,
        output=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monkey_date",
        dest="monkey_date",
        type=str,
        default='2017_06_18_17_45_17',
        help='Date of model directory.')

    args = parser.parse_args()
    main(**vars(args))
