import os
import re
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


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
        ax1.imshow(im)
        plot_coordinates(ax1, y, 'r')
        plot_coordinates(ax1, yhat, 'b')
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
        num_files=4,
        output_file='monkey_mosaic.png'
        ):
    monkey_dir = '/media/data_cifs/monkey_tracking/batches/predictions/New folder'
    ims = sorted(glob(os.path.join(monkey_dir, 'im_*')))[:num_files]
    im_list = []
    yhat_list = []
    ytrue_list = []
    for im in ims:
        file_name = re.search('\d+', im).group()
        images = np.load(im)
        yhats = np.load(os.path.join(monkey_dir, 'yhat_%s.npy' % file_name))
        ytrues = np.load(os.path.join(monkey_dir, 'ytrue_%s.npy' % file_name))
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