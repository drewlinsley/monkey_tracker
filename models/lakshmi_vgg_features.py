import numpy as np
import tensorflow as tf
import gc
from ops.loss_helper import flipped_gradient


def residual_wiring():
    """Wiring for residual connections."""
    l3_wiring = [{
        'pre': 'conv3_1',
        'post': 'conv3_3'
    }]
    l4_wiring = [{
        'pre': 'nconv4_1',
        'post': 'nconv4_3'
    }]
    l5_wiring = [{
        'pre': 'nconv5_1',
        'post': 'nconv5_3'
    }]
    return l3_wiring, l4_wiring, l5_wiring


class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(
            self,
            weight_npy_path=None,
            trainable=True,
            fine_tune_layers=None,
            model_optimizations=None):
        self.model_optimizations = model_optimizations
        if weight_npy_path is not None and self.model_optimizations['initialize_trained']:
            model_weights = weight_npy_path['vgg16']
            self.data_dict = np.load(model_weights, encoding='latin1').item()
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
            batchnorm=None,
            hr_fe_keys=['pool1', 'pool3', 'pool4', 'lrpool1', 'lrpool2']):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :if True, dropout will be turned on
        """

        self.input_bgr = tf.identity(rgb, name="lrp_input")
        if self.model_optimizations['dilated']:
            conv_op = 'dilated'
        else:
            conv_op = 'conv'
        if self.model_optimizations['skip'] == 'residual':
            l3_wiring, l4_wiring, l5_wiring = residual_wiring()
        elif self.model_optimizations['skip'] == 'dense':
            l3_wiring, l4_wiring, l5_wiring = residual_wiring()
        else:
            # Default no skips
            l3_wiring, l4_wiring, l5_wiring = None, None, None
        if self.model_optimizations['pool_v_stride'] == 'pool':
            layer_structure = [
                {
                    'layers': [conv_op, conv_op, 'pool'],
                    'weights': [64, 64, None],
                    'strides': [1, 1, None],
                    'names': ['conv1_1', 'conv1_2', 'pool1'],
                    'filter_size': [3, 3, 4]
                },
                {
                    'layers': [conv_op, conv_op, 'pool'],
                    'weights': [128, 128, None],
                    'strides': [1, 1, None],
                    'names': ['conv2_1', 'conv2_2', 'pool2'],
                    'filter_size': [3, 3, 4]
                },
                {
                    'layers': [conv_op, conv_op, conv_op, 'pool'],
                    'weights': [256, 256, 256, None],
                    'strides': [1, 1, 1, None],
                    'names': ['conv3_1', 'conv3_2', 'conv3_3', 'pool3'],
                    'filter_size': [3, 3, 3, 4],
                    'wiring': l3_wiring
                },
                {
                    'layers': [conv_op, conv_op, conv_op, 'pool'],
                    'weights': [512, 512, 512, None],
                    'strides': [1, 1, 1, None],
                    'names': ['nconv4_1', 'nconv4_2', 'nconv4_3', 'pool4'],
                    'filter_size': [3, 3, 3, 4],
                    'wiring': l4_wiring
                },
                {
                    'layers': [conv_op, conv_op, conv_op, 'pool'],
                    'weights': [512, 512, 512, None],
                    'strides': [1, 1, 1, None],
                    'names': ['nconv5_1', 'nconv5_2', 'nconv5_3', 'pool4'],
                    'filter_size': [3, 3, 3, 4],
                    'wiring': l5_wiring
                },
            ]
        else:
            layer_structure = [
                {
                    'layers': [conv_op, conv_op],
                    'weights': [64, 64],
                    'strides': [1, 4],
                    'names': ['conv1_1', 'conv1_2'],
                    'filter_size': [3, 3]
                },
                {
                    'layers': [conv_op, conv_op],
                    'weights': [128, 128],
                    'strides': [1, 4],
                    'names': ['conv2_1', 'conv2_2'],
                    'filter_size': [3, 3]
                },
                {
                    'layers': [conv_op, conv_op, conv_op],
                    'weights': [256, 256, 256],
                    'strides': [1, 1, 4],
                    'names': ['conv3_1', 'conv3_2', 'conv3_3'],
                    'filter_size': [3, 3, 3],
                    'wiring': l3_wiring
                },
                {
                    'layers': [conv_op, conv_op, conv_op],
                    'weights': [512, 512, 512],
                    'strides': [1, 1, 4],
                    'names': ['nconv4_1', 'nconv4_2', 'nconv4_3'],
                    'filter_size': [3, 3, 3],
                    'wiring': l4_wiring
                },
                {
                    'layers': [conv_op, conv_op, conv_op],
                    'weights': [512, 512, 512],
                    'strides': [1, 1, 4],
                    'names': ['nconv5_1', 'nconv5_2', 'nconv5_3'],
                    'filter_size': [3, 3, 3],
                    'wiring': l5_wiring
                },
            ]
        output = self.create_conv_tower(
            self.input_bgr,
            layer_structure,
            tower_name='highres_conv')
        self.data_dict = None
        return output

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
                for la, we, na, fs, ss in zip(
                        layer['layers'],
                        layer['weights'],
                        layer['names'],
                        layer['filter_size'],
                        layer['strides']):
                    if la == 'pool':
                        act = self.max_pool(
                            bottom=act,
                            name=na,
                            ksize=fs)
                    elif la == 'conv':
                        act = self.conv_layer(
                            bottom=act,
                            in_channels=int(act.get_shape()[-1]),
                            out_channels=we,
                            name=na,
                            stride=[1, ss, ss, 1],
                            filter_size=fs)
                    elif la == 'dilated':
                        act = self.dilated_conv_layer(
                            bottom=act,
                            in_channels=int(act.get_shape()[-1]),
                            out_channels=we,
                            name=na,
                            stride=[1, ss, ss, 1],
                            filter_size=fs)
                    elif la == 'res':
                        act = self.resnet_layer(
                            bottom=act,
                            layer_weights=we,
                            layer_name=na)
                    setattr(self, na, act)
                    print 'Added layer: %s' % na
                if 'wiring' in layer.keys() and layer['wiring'] is not None:
                    for wire in layer['wiring']:
                        self[wire['post']] += self[wire['pre']]
        return act

    def batchnorm(self, layer):
        m, v = tf.nn.moments(layer, [0])
        return tf.nn.batch_normalization(layer, m, v, None, None, 1e-3)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name, ksize=2):
        return tf.nn.max_pool(
            bottom, ksize=[1, ksize, ksize, 1],
            strides=[1, ksize, ksize, 1], padding='SAME', name=name)

    def conv_layer(
            self,
            bottom,
            in_channels,
            out_channels,
            name,
            filter_size=3,
            batchnorm=None,
            stride=[1, 1, 1, 1],
            nonlinearity=tf.nn.selu):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = nonlinearity(bias)

            if batchnorm is not None:
                if name in batchnorm:
                    relu = self.batchnorm(relu)

            return relu

    def dilated_conv_layer(
            self,
            bottom,
            in_channels,
            out_channels,
            name,
            filter_size=3,
            batchnorm=None,
            rate=4,
            stride=[1, 1, 1, 1],
            nonlinearity=tf.nn.selu):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)
            conv = tf.nn.atrous_conv2d(
                value=bottom,
                filters=filt,
                rate=rate,
                stride=stride,
                padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = nonlinearity(bias)

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
            self,
            initial_value,
            name,
            idx,
            var_name,
            in_size=None,
            out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            if name == 'conv1_1' and idx == 0:
                # Combine across feature channels if 1 channel input
                if int(self.input_bgr.get_shape()[-1]) == 1:
                    value = np.mean(
                        self.data_dict[name][idx], axis=-2, keepdims=True)
                elif int(self.input_bgr.get_shape()[-1]) != 3:
                    raise NotImplementedError('Weird input shape.')
                else:
                    value = self.data_dict[name][idx]
            else:
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
            else:
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
