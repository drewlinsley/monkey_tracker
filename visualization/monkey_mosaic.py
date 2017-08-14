import os
import re
import argparse
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from config import monkeyConfig
from ops import joint_list


def get_colors():
    joints = monkeyConfig().joint_names
    num_joints = len(joints)
    return cm.rainbow(np.linspace(0, 1, num_joints)), joints, num_joints


def save_mosaic(
        ims,
        yhats,
        ys,
        output=None,
        wspace=0.,
        hspace=0.,
        save_fig=True,
        conv_xy_to_hw=True):
    # Get a color for each yhat and a color for each ytrue
    colors, joints, num_joints = get_colors()
    rc = np.ceil(np.sqrt(len(ims))).astype(int)
    fig = plt.figure(figsize=(10, 10))
    gs1 = gridspec.GridSpec(rc, rc)
    lab_legend_artists = None
    for idx, (im, yhat) in enumerate(zip(ims, yhats)):
        if conv_xy_to_hw:
            yhat = yhat.reshape(-1, 2)[:, ::-1].reshape(1, -1)
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        # ax1.set_aspect('equal')
        # ax1.set_xlim([0, 240])
        # ax1.set_ylim([0, 320])
        ax1.imshow(np.log10(im), cmap='Greys_r')
        # ax1.imshow(im, cmap='Greys_r')
        if ys is not None:
            lab_legend_artists = plot_coordinates(
                ax1, ys[idx], colors, marker='.', markersize=1.5)
            leg_title = 'Predicted | True'
        else:
            lab_legend_artists = plot_coordinates(
                ax1, yhat, colors, marker='.', markersize=1.5)
            leg_title = 'Predicted position'
        est_legend_artists = plot_coordinates(
            ax1, yhat, colors,
            linestyle='none',
            markeredgewidth=.5,
            marker='o',
            mfc='none',
            markersize=1.5)
    plt.subplots_adjust(top=1, left=0)

    # Legend
    if lab_legend_artists is not None:
        patches = est_legend_artists + lab_legend_artists
    else:
        patches = est_legend_artists
    fig.legend(
        patches,
        (["" for _ in colors] +
         [j for j in joints]),
        title=leg_title,
        loc=1, frameon=False, numpoints=1, ncol=2,
        columnspacing=0, handlelength=0.25, markerscale=2)
    if save_fig:
        plt.savefig(output)
    else:
        print 'Showing not saving mosaic.'
        plt.show()
        plt.close('fig')


def save_3d_mosaic(
        ims,
        pxs,
        pys,
        pzs,
        ys,
        output=None,
        wspace=0.,
        hspace=0.,
        save_fig=True):
    # Get a color for each yhat and a color for each ytrue
    colors, joints, num_joints = get_colors()
    rc = np.ceil(np.sqrt(len(ims))).astype(int)
    fig = plt.figure(figsize=(10, 10))
    gs1 = gridspec.GridSpec(rc, rc)
    lab_legend_artists = None
    joint_labels = joint_list.joint_data()
    for idx, (im, ixs, iys, izs) in enumerate(zip(ims, pxs, pys, pzs)):
        ax1 = plt.subplot(gs1[idx], projection='3d')
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        # ax1.set_aspect('equal')
        # ax1.set_xlim([0, 240])
        # ax1.set_ylim([0, 320])
        # ax1.imshow(im, cmap='Greys_r')
        if ys is not None:
            lab_legend_artists = plot_coordinates(
                ax1, ys[idx], colors, marker='.', markersize=1.5)
            leg_title = 'Predicted | True'
        else:
            leg_title = 'Predicted'
        est_legend_artists = ax1.scatter(
                ixs,
                iys,
                izs,
                c=colors,
                marker='o')
        # Construct skeleton
        jxs, jys, jzs = [], [], []
        for jc in joint_labels.joint_connections:
            jidx = joint_labels.joint_names.index(jc)
            jxs += [ixs[jidx]]
            jys += [iys[jidx]]
            jzs += [izs[jidx]]
        ax1.plot(ixs, iys, izs)
        ax1.view_init(0, 20)
    plt.subplots_adjust(top=1, left=0)

    # Legend
    if lab_legend_artists is not None:
        patches = est_legend_artists + lab_legend_artists
    else:
        patches = est_legend_artists
    # fig.legend(
    #     patches,
    #     (["" for _ in colors] +
    #      [j for j in joints]),
    #     title=leg_title,
    #     loc=1, frameon=False, numpoints=1, ncol=2,
    #     columnspacing=0, handlelength=0.25, markerscale=2)
    if save_fig:
        plt.savefig(output)
    else:
        print 'Showing not saving mosaic.'
        plt.show()
        plt.close('fig')


def xyz_vector_to_xy(vector, num_dims=2):
    return vector.reshape(-1, num_dims)[:, :2]


def plot_coordinates(ax, vector, colors, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    xy = xyz_vector_to_xy(vector)
    if colors is None:
        colors = ['r'] * xy.shape[0]
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
        max_ims=32
        ):

    monkey_dir = os.path.join(dmurphy_npy_dir, monkey_date)
    # Eventually read settings from the saved config file
    if unnormalize:
        # Because normalization was fucked up in the training script
        config = monkeyConfig()
        unnormalize_vec = np.asarray(
            config.image_target_size[:config.num_dims] + [config.max_depth]).repeat(
            len(config.joint_order))
        normalize_vec = np.asarray(
            config.image_target_size[:config.num_dims] + [config.max_depth]).reshape(
            1, -1).repeat(23, axis=0).reshape(1, -1)
    if normalize:
        config = monkeyConfig()
        normalize_vec = np.asarray(
            config.image_target_size[:config.num_dims] + [config.max_depth]).reshape(
            1, -1).repeat(23, axis=0).reshape(1, -1)
    ims = glob(os.path.join(monkey_dir, 'im_*'))
    ims.sort(key=os.path.getmtime)
    ims = ims[::-1][:num_files]
    im_list = []
    yhat_list = []
    ytrue_list = []
    val_images_list = []
    val_yhats_list = []
    for im in ims:
        file_name = re.search('\d+', im.split('/')[-1]).group()

        # Train images
        images = np.squeeze(np.load(im))
        yhats = np.load(os.path.join(monkey_dir, 'yhat_%s.npy' % file_name))
        ytrues = np.load(os.path.join(monkey_dir, 'ytrue_%s.npy' % file_name))

        # Validation images
        val_npz = os.path.join(monkey_dir, '%s_val_coors.npz' % file_name)
        if os.path.exists(val_npz):
            val_data = np.load(val_npz)
            normalize_vec = val_data['normalize_vec']
            val_images = val_data['val_ims']
            val_yhats = val_data['val_pred']
            val_yhats *= normalize_vec
            run_vals = True
        else:
            run_vals = False

        if normalize:
            yhats *= normalize_vec
            ytrues *= normalize_vec
        if unnormalize:
            yhats /= unnormalize_vec
            ytrues /= unnormalize_vec
            val_yhats /= unnormalize_vec
            yhats *= normalize_vec
            ytrues *= normalize_vec
            val_yhats *= normalize_vec
        [im_list.append(x) for x in images]
        [yhat_list.append(x) for x in yhats]
        [ytrue_list.append(x) for x in ytrues]
        [val_images_list.append(x) for x in val_images]
        [val_yhats_list.append(x) for x in val_yhats]

    if max_ims is not None:
        rand_order = np.random.permutation(len(im_list))
        im_list = np.asarray(im_list)[rand_order][:max_ims]
        yhat_list = np.asarray(yhat_list)[rand_order][:max_ims]
        ytrue_list = np.asarray(ytrue_list)[rand_order][:max_ims]
        rand_order = np.random.permutation(len(val_images_list))
        val_images_list = np.asarray(val_images_list)[rand_order][:max_ims]
        val_yhats_list = np.asarray(val_yhats_list)[rand_order][:max_ims]
    save_mosaic(
        ims=im_list,
        yhats=yhat_list,
        ys=ytrue_list,
        output=output_file)
    if run_vals:
        save_mosaic(
            ims=val_images_list[:, :, :, 0],
            yhats=val_yhats_list,
            ys=np.copy(val_yhats_list),  # np.zeros_like(val_yhats_list),
            output='val_%s' % output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        dest="monkey_date",
        type=str,
        default='cnn_multiscale_high_res_low_res_skinny_pose_occlusion_2017_06_22_12_44_05', # 2017_06_18_11_42_34
        help='Date of model directory.')

    args = parser.parse_args()
    main(**vars(args))
