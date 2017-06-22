import re
import os
import numpy as np
import tensorflow as tf
from glob import glob
from math import sqrt


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def fine_tune_prepare_layers(tf_vars, finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        ss = [v.name.find(x) != -1 for x in finetune_vars]
        if True in ss:
            ft_vars.append(v)
        else:
            other_vars.append(v)
    return other_vars, ft_vars


def ft_optimizer_list(cost, opt_vars, optimizer, lrs, grad_clip=False):
    """Efficient optimization for fine tuning a net."""
    ops = []
    gvs = []
    for v, l in zip(opt_vars, lrs):
        if grad_clip:
            optim = optimizer(l)
            gvs = optim.compute_gradients(cost, var_list=v)
            capped_gvs = [
                (tf.clip_by_norm(grad, 10.), var)
                if grad is not None else (grad, var) for grad, var in gvs]
            ops.append(optim.apply_gradients(capped_gvs))
        else:
            ops.append(optimizer(l).minimize(cost, var_list=v))
    return tf.group(*ops), gvs


def ft_optimized(cost, var_list_1, var_list_2, optimizer, lr_1, lr_2):
    """Applies different learning rates to specified layers."""
    opt1 = optimizer(lr_1)
    opt2 = optimizer(lr_2)
    grads = tf.gradients(cost, var_list_1 + var_list_2)
    grads1 = grads[:len(var_list_1)]
    grads2 = grads[len(var_list_1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list_1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list_2))
    return tf.group(train_op1, train_op2)


def ft_non_optimized(cost, other_opt_vars, ft_opt_vars, optimizer, lr_1, lr_2):
    op1 = tf.train.AdamOptimizer(lr_1).minimize(cost, var_list=other_opt_vars)
    op2 = tf.train.AdamOptimizer(lr_2).minimize(cost, var_list=ft_opt_vars)
    return tf.group(op1, op2)  # ft_optimize is more efficient.


def class_accuracy(pred, targets):
    return tf.reduce_mean(
        tf.to_float(
            tf.equal(tf.argmax(pred, 1), tf.cast(
                targets, dtype=tf.int64))))  # assuming targets is an index


def count_nonzero(data):
    return tf.reduce_sum(tf.cast(tf.not_equal(data, 0), tf.float32))


def fscore(pred, targets):
    predicted = tf.cast(tf.argmax(pred, axis=1), tf.int32)

    # Count true +, true -, false + and false -.
    tp = count_nonzero(predicted * targets)
    # tn = tf.count_nonzero((predicted - 1) * (targets - 1))
    fp = count_nonzero(predicted * (targets - 1))
    fn = count_nonzero((predicted - 1) * targets)

    # Calculate accuracy, precision, recall and F1 score.
    # accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    return fmeasure


def zscore(x):
    mu, std = tf.nn.moments(x, axes=[0])
    return (x - mu) / std


def correlation(x, y):
    return tf.contrib.metrics.streaming_pearson_correlation(
        predictions=x,
        labels=y,
        name="pearson")


def tf_confusion_matrix(pred, targets):
    return tf.contrib.metrics.confusion_matrix(pred, targets)


def softmax_cost(logits, labels, ratio=None, label_reshape=None):
    if label_reshape is not None:
        # Have to reshape the labels to match output of MLP
        labels = tf.reshape(labels, label_reshape)

    if ratio is not None:
        # ratios = tf.get_variable(
        # name='ratio', initializer=[1.0 - ratio, ratio])
        ratios = tf.get_variable(
            name='ratio', initializer=ratio[::-1])[None, :]
        # weighted_logits = tf.mul(logits, ratios)
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(
        # weighted_logits, labels)
        weights_per_label = tf.matmul(
            tf.one_hot(labels, 2), tf.transpose(tf.cast(ratios, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels)))
        # return tf.reduce_mean(
        # tf.contrib.losses.sparse_softmax_cross_entropy(
        # logits, labels, weight=ratio))
    else:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))


def find_ckpts(config, dirs=None):
    if dirs is None:
        dirs = sorted(
            glob(
                config.train_checkpoint + config.which_dataset + '*'),
            reverse=True)[0]  # only the newest model run
    ckpts = sorted(glob(dirs + '/*.ckpt*'))
    ckpts = [ck for ck in ckpts if '.meta' not in ck]  # Don't include metas
    ckpt_names = [re.split('-', ck)[-1] for ck in ckpts]
    return np.asarray(ckpts), np.asarray(ckpt_names)


def l1_loss(yhat, y):
    return tf.reduce_sum(tf.abs(yhat - y), axis=-1)


def l2_loss(yhat, y):
    return tf.reduce_mean(tf.pow(yhat - y, 2), axis=-1)


def thomas_l1_loss(model, train_data_dict, config):
    if config.selected_joints is None:
        use_joints = config.joint_order
    else:
        use_joints = config.selected_joints

    if config.loss_type == 'l1':
        loss = l1_loss
    elif config.loss_type == 'l2':
        loss = l2_loss
    else:
        raise RuntimeError(
            'Cannot understand your selected loss type. Needs to be implemented?')

    res_gt = tf.reshape(
        train_data_dict['label'],
        [config.train_batch, len(use_joints), config.keep_dims])
    res_pred = [tf.reshape(
        model[x],
        [
            config.train_batch,
            len(use_joints),
            config.keep_dims]
            ) for x in model.joint_label_output_keys]
    label_losses = [loss(x, res_gt) for x in res_pred]

    # Return variance for tf.summary
    loss_x_batch = [tf.reduce_sum(x, axis=0) for x in label_losses]
    for lx in loss_x_batch:
        [tf.summary.scalar(
            '%s' % la, lo[0]) for la, lo in zip(
            use_joints, tf.split(lx, len(use_joints)))]
    joint_variance = []
    for idx, lb in enumerate(loss_x_batch):
        _, iv = tf.nn.moments(lb, [0])
        joint_variance += [iv]
        tf.summary.scalar('Joint variance %s' % idx, iv)

    # Prepare loss for output
    label_loss = tf.add_n([tf.reduce_mean(x) for x in label_losses])
    return label_loss, use_joints, joint_variance


def get_normalization_vec(config, num_joints):
    normalize_values = np.asarray(
        config.image_target_size[:2] + [
            config.max_depth])[:config.keep_dims]
    if len(normalize_values) == 1:
        normalize_values = normalize_values[None, :]
    return normalize_values.reshape(
        1, -1).repeat(num_joints, axis=0).reshape(1, -1)


def put_kernels_on_grid(kernel, pad=1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
          if n % i == 0:
            if i == 1: print('Who would enter a prime number of filters')
            return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x

