import numpy as np
import tensorflow as tf


VGG_MEAN = [103.939, 116.779, 123.68]


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
                self, vgg16_npy_path=None, trainable=True,
                fine_tune_layers=None):
        if vgg16_npy_path is not None:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            # pop the specified keys from the weights that will be loaded
            if fine_tune_layers is not None:
                for key in fine_tune_layers:
                    del self.data_dict[key]
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(
            self, rgb, output_categories=None,
            train_mode=None, batchnorm=None,
            resize_layer='pool3'):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """
        if output_categories is None:
            output_categories = 21  # len(config.labels)

        rgb_scaled = rgb * 255.0  # Scale up to imagenet's uint8

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # VGG16 starts here
        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(
            self.pool4, 512, 512, "conv5_1", batchnorm)
        self.conv5_2 = self.conv_layer(
            self.conv5_1, 512, 512, "conv5_2", batchnorm)
        self.conv5_3 = self.conv_layer(
            self.conv5_2, 512, 512, "conv5_3", batchnorm)
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        # Consider changing these to numpy conditionals
        if train_mode is not None:
            self.relu6 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)
        if batchnorm is not None:
            if 'fc6' in batchnorm:
                self.relu6 = self.batchnorm(self.relu6)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)
        if batchnorm is not None:
            if 'fc7' in batchnorm:
                self.relu7 = self.batchnorm(self.relu7)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")
        if batchnorm is not None:
            if 'fc8' in batchnorm:
                self.fc8 = self.batchnorm(self.fc8)

        self.cnn_prob = tf.nn.softmax(self.fc8, name="cnn_prob")

        # Get vgg16 features
        resize_layer_dims = tf.cast(
            self.data_dict[resize_layer][1].get_shape(), tf.int32)
        l1 = tf.image.resize_images(
            self.pool3, resize_layer_dims)
        l2 = tf.image.resize_image_with_crop_or_pad(
            self.pool4, resize_layer_dims)
        l3 = tf.image.resize_image_with_crop_or_pad(
            self.pool5, resize_layer_dims)
        self.feature_encoder = tf.concat([l1, l2, l3], axis=3)  # channelwise

        # Add 1x1 convs to vgg features
        # Conv 1 - 1x1/relu/dropout/batchnorm
        self.fc_conv1 = self.conv_layer(
            self.feature_encoder,
            int(self.feature_encoder.get_shape()[-1]), 32,
            "fc_conv1", filter_size=1)
        self.fc_relu1 = tf.nn.relu(self.fc_conv1)

        if train_mode is not None:
            # Add dropout during training
            self.fc_relu1 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.fc_relu1, 0.5), lambda: self.fc_relu1)

        if batchnorm is not None:
            if 'fc_conv1' in batchnorm:
                self.fc_relu1 = self.batchnorm(self.fc_relu1)

        # Conv 2 - 1x1/relu/dropout/batchnorm
        self.fc_onv2 = self.conv_layer(
            self.relu1, int(self.fc_relu1.get_shape()[-1]), 64,
            "fc_conv2", filter_size=1)
        self.fc_relu2 = tf.nn.relu(self.fc_conv2)

        if train_mode is not None:
            # Add dropout during training
            self.fc_relu2 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.fc_relu2, 0.5), lambda: self.fc_relu2)

        if batchnorm is not None:
            if 'fc_conv2' in batchnorm:
                self.fc_relu2 = self.batchnorm(self.fc_relu2)

        # Conv 3 - 1x1/relu/dropout/batchnorm
        self.fc_conv3 = self.conv_layer(
            self.fc_relu2, int(
                self.fc_relu2.get_shape()[-1]), output_categories * 2,
            "fc_conv3", filter_size=1)
        self.fc_relu3 = tf.nn.relu(self.fc_conv3)

        if train_mode is not None:
            # Add dropout during training
            self.fc_relu3 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.fc_relu3, 0.5), lambda: self.fc_relu3)

        if batchnorm is not None:
            if 'fc_conv3' in batchnorm:
                self.fc_relu3 = self.batchnorm(self.fc_relu3)

        # image-sized output
        self.logits = self.conv_layer(
            self.fc_relu3, int(
                self.fc_relu3.get_shape()[-1]), output_categories,
            "fc_conv4", filter_size=1)

        # normalize over filter dimensions
        logit_shape = tf.cast(self.logits.get_shape(), tf.int32)

        # Because I'm running with TF v.10, I have to reshape then apply
        # softmax then reshape instead of using tf.nn.softmax(..., dim=-1)
        self.res_logits = tf.reshape(
            self.logits,
            [logit_shape[0], logit_shape[1] * logit_shape[2],
                logit_shape[3]])  # bsXpixelsX21
        self.prob = self.batchsoftmax(self.res_logits, name="prob")

        # Finishing touches
        self.data_dict = None

    def batchsoftmax(self, layer, name=None, axis=2):
        exp_layer = tf.exp(layer)
        exp_sums = tf.expand_dims(
            tf.reduce_sum(exp_layer, reduction_indices=[axis]), dim=axis)
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
                    out_channels, name, batchnorm=None):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                3, in_channels, out_channels, name)

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

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
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
