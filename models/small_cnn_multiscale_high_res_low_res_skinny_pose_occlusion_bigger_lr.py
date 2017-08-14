import numpy as np
import tensorflow as tf
import gc
from ops.loss_helper import flipped_gradient


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
                self, weight_npy_path=None, trainable=True,
                fine_tune_layers=None):
        if weight_npy_path is not None:
            self.data_dict = np.load(weight_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.joint_label_output_keys = []

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(
            self,
            rgb,
            target_variables=None,
            train_mode=None,
            batchnorm=None,
            hr_fe_keys=['pool1', 'pool3', 'pool4', 'lrpool1', 'lrpool2'],
            ):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """

        if 'label' in target_variables.keys():
            if len(target_variables['label'].get_shape()) == 1:
                output_shape = 1
            else:
                output_shape = int(
                    target_variables['label'].get_shape()[-1])

        if 'occlusion' in target_variables.keys():
            if len(target_variables['occlusion'].get_shape()) == 1:
                occlusion_shape = 1
            else:
                occlusion_shape = int(
                    target_variables['occlusion'].get_shape()[-1])

        if 'z' in target_variables.keys():
            z_shape = int(
                target_variables['z'].get_shape()[-1])

        if 'size' in target_variables.keys():
            size_shape = int(
                target_variables['size'].get_shape()[-1])

        if 'pose' in target_variables.keys():
            if len(target_variables['pose'].get_shape()) == 1:
                pose_shape = 1
            else:
                pose_shape = int(
                    target_variables['pose'].get_shape()[-1])

        if 'domain_adaptation' in target_variables.keys():
            domain_shape = 2  # Always binary

        input_bgr = tf.identity(rgb, name="lrp_input")
        layer_structure = [
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [32, 32, None],
                'names': ['conv1_1', 'conv1_2', 'pool1'],
                'filter_size': [5, 5, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [32, 32, None],
                'names': ['conv2_1', 'conv2_2', 'pool2'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [64, 64, None],
                'names': ['conv3_1', 'conv3_2', 'pool3'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [64, 64, None],
                'names': ['conv4_1', 'conv4_2', 'pool4'],
                'filter_size': [3, 3, None]
            }]

        self.create_conv_tower(
            input_bgr,
            layer_structure,
            tower_name='highres_conv')

        # Replace this lowres tower with atrous convolutions
        rescaled_shape = [(
            int(x) - 1) // 2 + 1 for x in input_bgr.get_shape()[1:3]]
        res_input_bgr = tf.image.resize_bilinear(input_bgr, rescaled_shape)

        layer_structure = [
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [32, 32, None],
                'names': ['lrconv1_1', 'lrconv1_2', 'lrpool1'],
                'filter_size': [5, 5, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [32, 32, None],
                'names': ['lrconv2_1', 'lrconv2_2', 'lrpool2'],
                'filter_size': [3, 3, None]
            }]

        self.create_conv_tower(
            res_input_bgr,  # pass input_bgr instead of this
            layer_structure,
            tower_name='lowres_conv')

        # Rescale feature maps to the largest in the hr_fe_keys
        resize_h = np.max([int(self[k].get_shape()[1]) for k in hr_fe_keys])
        resize_w = np.max([int(self[k].get_shape()[2]) for k in hr_fe_keys])
        new_size = np.asarray([resize_h, resize_w])
        high_fe_layers = [self.batchnorm(
            tf.image.resize_bilinear(
                self[x], new_size)) for x in hr_fe_keys]
        self.high_feature_encoder = tf.concat(high_fe_layers, 3)

        # High-res 1x1 X 2
        self.high_feature_encoder_1x1_0 = self.conv_layer(
            self.high_feature_encoder,
            int(self.high_feature_encoder.get_shape()[-1]),
            256,
            "high_feature_encoder_1x1_0",
            filter_size=1)
        if train_mode is not None:
            self.high_feature_encoder_1x1_0 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.high_feature_encoder_1x1_0, 0.5),
                lambda: self.high_feature_encoder_1x1_0)
        self.high_1x1_0_pool = self.max_pool(
            self.high_feature_encoder_1x1_0,
            'high_1x1_0_pool')

        self.high_feature_encoder_1x1_1 = self.conv_layer(
            self.high_1x1_0_pool,
            int(self.high_1x1_0_pool.get_shape()[-1]),
            192,
            "high_feature_encoder_1x1_1",
            filter_size=1)
        if train_mode is not None:
            self.high_feature_encoder_1x1_1 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(
                    self.high_feature_encoder_1x1_1, 0.5),
                lambda: self.high_feature_encoder_1x1_1)
        self.high_1x1_1_pool = self.max_pool(
            self.high_feature_encoder_1x1_1,
            'high_1x1_1_pool')

        self.high_feature_encoder_1x1_2 = self.conv_layer(
            self.high_1x1_1_pool,
            int(self.high_1x1_1_pool.get_shape()[-1]),
            128,
            "high_feature_encoder_1x1_2",
            filter_size=1)
        if train_mode is not None:
            self.high_feature_encoder_1x1_2 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.high_feature_encoder_1x1_2, 0.5),
                lambda: self.high_feature_encoder_1x1_2)
        self.high_1x1_2_pool = tf.contrib.layers.flatten(
            self.max_pool(self.high_feature_encoder_1x1_2, 'high_1x1_2_pool'))

        if 'label' in target_variables.keys():
            self.output = self.fc_layer(
                self.high_1x1_2_pool,
                int(self.high_1x1_2_pool.get_shape()[-1]),
                output_shape,
                'output')
            self.joint_label_output_keys += ['output']

        if 'size' in target_variables.keys():
            self.size = self.fc_layer(
                self.high_1x1_2_pool,
                int(self.high_1x1_2_pool.get_shape()[-1]),
                size_shape,
                'size')

        if 'z' in target_variables.keys():
            # z-dim head -- label + activations
            self.z_scores = tf.squeeze(
                    self.fc_layer(
                        self.high_1x1_2_pool,
                        int(self.high_1x1_2_pool.get_shape()[-1]),
                        z_shape,
                        'z_sc')
                )
            out_z = tf.concat([self.z_scores, self.output], axis=1)
            self.z = tf.squeeze(
                    self.fc_layer(
                        out_z,
                        int(out_z.get_shape()[-1]),
                        z_shape,
                        'z')
                )
            self.joint_label_output_keys += ['z']

        if 'occlusion' in target_variables.keys():
            # Occlusion head
            self.occlusion = tf.squeeze(
                    self.fc_layer(
                        self.high_1x1_2_pool,
                        int(self.high_1x1_2_pool.get_shape()[-1]),
                        occlusion_shape,
                        "occlusion")
                    )

        if 'pose' in target_variables.keys():
            # Occlusion head
            self.pose = tf.squeeze(
                    self.fc_layer(
                        self.high_1x1_2_pool,
                        int(self.high_1x1_2_pool.get_shape()[-1]),
                        pose_shape,
                        "pose")
                )

        if 'domain_adaptation_flip' in target_variables.keys():
            # Domain adaptation head
            bottom = flipped_gradient(
                tf.contrib.layers.flatten(self.high_1x1_0_pool))
            self.domain_adaptation_fc = tf.squeeze(
                    self.fc_layer(
                        bottom,
                        int(bottom.get_shape()[-1]),
                        32,
                        "domain_adaptation_fc")
                )
            self.domain_adaptation = tf.squeeze(
                    self.fc_layer(
                        self.domain_adaptation_fc,
                        int(self.domain_adaptation_fc.get_shape()[-1]),
                        domain_shape,
                        "domain_adaptation")
                )

        self.data_dict = None

    def resnet_layer(
            self,
            bottom,
            layer_weights,
            layer_name):
        ln = '%s_branch' % layer_name
        in_layer = self.conv_layer(
                bottom,
                int(bottom.get_shape()[-1]),
                layer_weights[0],
                batchnorm=[ln],
                name=ln)
        branch_conv = tf.identity(in_layer)
        for idx, lw in enumerate(layer_weights):
            ln = '%s_%s' % (layer_name, idx)
            in_layer = self.conv_layer(
                in_layer,
                int(in_layer.get_shape()[-1]),
                lw,
                batchnorm=[ln],
                name=ln)
        return branch_conv + in_layer

    def create_conv_tower(self, act, layer_structure, tower_name):
        print 'Creating tower: %s' % tower_name
        with tf.variable_scope(tower_name):
            for layer in layer_structure:
                for la, we, na, fs in zip(
                        layer['layers'],
                        layer['weights'],
                        layer['names'],
                        layer['filter_size']):
                    if la == 'pool':
                        act = self.max_pool(
                            bottom=act,
                            name=na)
                    elif la == 'conv':
                        act = self.conv_layer(
                            bottom=act,
                            in_channels=int(act.get_shape()[-1]),
                            out_channels=we,
                            name=na,
                            filter_size=fs
                        )
                    elif la == 'res':
                        act = self.resnet_layer(
                            bottom=act,
                            layer_weights=we,
                            layer_name=na)
                    setattr(self, na, act)
                    print 'Added layer: %s' % na
        return act

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
