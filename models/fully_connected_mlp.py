import numpy as np
import tensorflow as tf


class model_struct:
    """
    A trainable 1x1 conv model.
    """

    def __init__(
                self, trainable=True):
        self.data_dict = None
        self.var_dict = {}
        self.trainable = trainable

    def build(
            self, features, output_categories=None,
            train_mode=None, batchnorm=None):
        """
        Param train_mode: a bool tensor, usually a placeholder:
        if True, dropout will be turned on
        Model structure goes: Expand/Expand/Compress/Compress
        """
        if output_categories is None:
            output_categories = 21  # len(config.labels)

        # Conv 1 - 1x1/relu/dropout/batchnorm
        self.fc1 = self.fc_layer(
            features, int(features.get_shape()[-1]), 128, "fc1")
        self.relu1 = tf.nn.relu(self.fc1)

        if train_mode is not None:
            # Add dropout during training
            self.relu1 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.relu1, 0.5), lambda: self.relu1)

        if batchnorm is not None:
            if 'fc1' in batchnorm:
                self.relu1 = self.batchnorm(self.relu1)

        # Conv 2 - 1x1/relu/dropout/batchnorm
        self.fc2 = self.fc_layer(
            self.relu1, int(self.relu1.get_shape()[-1]), 256, "fc2")
        self.relu2 = tf.nn.relu(self.fc2)

        if train_mode is not None:
            # Add dropout during training
            self.relu2 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.relu2, 0.5), lambda: self.relu2)

        if batchnorm is not None:
            if 'fc2' in batchnorm:
                self.relu2 = self.batchnorm(self.relu2)

        # Conv 3 - 1x1/relu/dropout/batchnorm
        self.fc3 = self.fc_layer(
            self.relu2, int(self.relu2.get_shape()[-1]), output_categories * 2,
            "fc3")
        self.relu3 = tf.nn.relu(self.fc3)

        if train_mode is not None:
            # Add dropout during training
            self.relu3 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.relu3, 0.5), lambda: self.relu3)

        if batchnorm is not None:
            if 'fc3' in batchnorm:
                self.relu3 = self.batchnorm(self.relu3)

        # image-sized output
        self.res_logits = self.fc_layer(
            self.relu3, int(self.relu3.get_shape()[-1]), output_categories,
            "fc4")

        # normalize over filter dimensions
        self.prob = tf.nn.softmax(self.res_logits, name="prob")
        self.data_dict = None

    def batchsoftmax(self, layer, name=None, axis=2):
        exp_layer = tf.exp(layer)
        exp_sums = tf.expand_dims(
            tf.reduce_sum(exp_layer, axis=[axis]), axis=axis)
        return tf.div(exp_layer, exp_sums, name=name)

    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(
                    self, bottom, in_channels,
                    out_channels, name, filter_size=3, batchnorm=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolian to numpy
            var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./fully_connect_weights.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if name not in data_dict.keys():
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
