import numpy as np
import tensorflow as tf
import gc


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
                self, vgg16_npy_path=None, trainable=True,
                fine_tune_layers=None):
        if vgg16_npy_path is not None: 
            print 'Ignoring vgg16_npy_path (not using a vgg!).'
        self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.VGG_MEAN = [103.939, 116.779, 123.68]
 
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name): 
        return hasattr(self, name)

    def build(
            self,
            rgb,
            occlusions=True,
            output_shape=None,
            train_mode=None,
            batchnorm=None,
            fe_keys=None,
            hr_fe_keys=['pool2', 'pool3', 'pool4', 'pool5'],
            lr_fe_keys=['lr_pool2', 'lr_pool3']
            ):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """
        if fe_keys is not None:
            print 'I see you supplied feature extractor keys... These are being ignored.'
        if output_shape is None: 
            output_shape = 1
        if occlusions is not None:
            occlusion_shape = output_shape // 3

        # rgb_scaled = rgb * 255.0  # Scale up to imagenet's uint8

        # # Convert RGB to BGR
        # red, green, blue = tf.split(rgb_scaled, 3, 3)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # bgr = tf.concat([
        #     blue - self.VGG_MEAN[0],
        #     green - self.VGG_MEAN[1],
        #     red - self.VGG_MEAN[2],
        # ], 3, name='bgr')
        # bgr = tf.split(rgb, 3, 3)[0]

        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        input_bgr = tf.identity(rgb, name="lrp_input")
        # Main Head
        self.conv1_1 = self.conv_layer(input_bgr, int(input_bgr.get_shape()[-1]), 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 128, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 128, 128, "conv3_2")
        self.pool3 = self.max_pool(self.conv3_2, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 128, 256, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 256, 256, "conv4_2")
        self.pool4 = self.max_pool(self.conv4_2, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, int(self.pool4.get_shape()[-1]), 256, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, int(self.conv5_1.get_shape()[-1]), 256, "conv5_2")
        self.pool5 = self.max_pool(self.conv5_2, 'pool5')

        # High-res feature encoder
        resize_size = [int(x) for x in self[hr_fe_keys[np.argmax(
            [int(self[x].get_shape()[-1]) for x in hr_fe_keys])]].get_shape()]
        new_size = np.asarray([resize_size[1], resize_size[2]])

        high_fe_layers = [self.batchnorm(
            tf.image.resize_bilinear(
                self[x], new_size)) for x in hr_fe_keys]
        self.high_feature_encoder = tf.concat(high_fe_layers, 3)

        # High-res 1x1 X 2
        self.high_feature_encoder_1x1_0 = self.conv_layer(
            self.high_feature_encoder,
            int(self.high_feature_encoder.get_shape()[-1]),
            512,
            "high_feature_encoder_1x1_0",
            filter_size=1)
        if train_mode is not None:
            self.high_feature_encoder_1x1_0 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.high_feature_encoder_1x1_0, 0.5), lambda: self.high_feature_encoder_1x1_0)
        self.high_1x1_0_pool = self.max_pool(self.high_feature_encoder_1x1_1, 'high_1x1_0_pool')

        self.high_feature_encoder_1x1_1 = self.conv_layer(
            self.high_1x1_0_pool,
            int(self.high_1x1_0_pool.get_shape()[-1]),
            512,
            "high_feature_encoder_1x1_1",
            filter_size=1)
        if train_mode is not None:
            self.high_feature_encoder_1x1_1 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.high_feature_encoder_1x1_1, 0.5), lambda: self.high_feature_encoder_1x1_1)
        self.high_1x1_1_pool = self.max_pool(self.high_feature_encoder_1x1_1, 'high_1x1_1_pool')        

        self.high_feature_encoder_1x1_2 = self.conv_layer(
            self.high_1x1_1_pool,
            int(self.high_1x1_1_pool.get_shape()[-1]),
            256,
            "high_feature_encoder_1x1_2",
            filter_size=1)
        if train_mode is not None:
            self.high_feature_encoder_1x1_2 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.high_feature_encoder_1x1_2, 0.5), lambda: self.high_feature_encoder_1x1_2)
        self.high_1x1_2_pool = tf.contrib.layers.flatten(
            self.max_pool(self.high_feature_encoder_1x1_2, 'high_1x1_2_pool'))
        self.fc8 = self.fc_layer(
            self.high_1x1_2_pool,
            int(self.high_1x1_2_pool.get_shape()[-1]),
            output_shape,
            "fc8") 

        if occlusions is not None:
            # Occlusion head
            self.fc8_occlusion = self.fc_layer(
                self.high_1x1_2_pool,
                int(self.high_1x1_2_pool.get_shape()[-1]),
                occlusion_shape,
                "fc8_occlusion_scores")
        self.data_dict = None

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
                    out_channels, name, filter_size=3, batchnorm=None, stride=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
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

    def get_conv_var(
            self, filter_size, in_channels, out_channels,
            name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [filter_size, filter_size, in_channels, out_channels],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [filter_size, filter_size, in_channels, out_channels],
                0.0, 0.001)
        bias_init = tf.truncated_normal([out_channels], .0, .001)
        filters = self.get_var(weight_init, name, 0, name + "_filters")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolian to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1])
            else:
                var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}
        num_files = 0

        for (name, idx), var in self.var_dict.items():
            # print(var.get_shape())
            if name == 'fc6':
                np.save(npy_path+str(num_files), data_dict)
                data_dict.clear()
                gc.collect()
                num_files += 1
                for i, item in enumerate(tf.split(var, 8, 0)):
                    # print(i)
                    name = 'fc6-'+ str(i)
                    if name not in data_dict.keys():
                        data_dict[name] = {}
                    data_dict[name][idx] = sess.run(item)
                    np.save(npy_path+str(num_files), data_dict)
                    data_dict.clear()
                    gc.collect()
                    num_files += 1
            else :
                var_out = sess.run(var)
                if name not in data_dict.keys():
                    data_dict[name] = {}
                data_dict[name][idx] = var_out

        np.save(npy_path+str(num_files), data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
