import os
import re
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from config import monkeyConfig


def save_mosaic(ims, yhats, ys, output):
    rc = np.ceil(np.sqrt(len(ims))).astype(int)
    plt.figure(figsize=(20, 20))
    gs1 = gridspec.GridSpec(rc, rc)
    gs1.update(wspace=0.0001, hspace=0.0001)  # set the spacing between axes.
    for idx, (im, yhat, y) in enumerate(zip(ims, yhats, ys)):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(np.log10(im), cmap='Greys_r')
        plot_coordinates(ax1, y, 'r')
        plot_coordinates(ax1, yhat, 'b')
    plt.subplots_adjust(top=1)
    plt.suptitle('Blue = True Joint Position\nBlack = Predicted Joint Position.')
    plt.savefig(output)


def xyz_vector_to_xy(vector):
    return vector.reshape(-1, 3)[:, :2]


def plot_coordinates(ax, vector, color):
    xy = xyz_vector_to_xy(vector)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=color,
        marker='.',
        s=15,
        edgecolors='face')  # , alpha=0.5)
    return ax


def main(
        num_files=1,
        output_file='monkey_mosaic.png',
        monkey_dir='/media/data_cifs/monkey_tracking/batches/test/2017_06_14_14_48_44',
        normalize=False,
        unnormalize=False
        ):

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

    save_mosaic(
        ims=im_list,
        yhats=yhat_list,
        ys=ytrue_list,
        output=output_file)


if __name__ == '__main__':
    main()
