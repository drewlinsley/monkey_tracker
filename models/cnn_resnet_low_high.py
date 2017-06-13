import numpy as np
import tensorflow as tf

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)



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
            output_shape=None,
            train_mode=None,
            batchnorm=None,
            fe_keys=['pool2', 'pool3', 'pool4'],
            mask_head='pool4'
            ):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """
        if output_shape is None:
            output_shape = 1

        # Split off a single channel from our "rgb" depth image
        # bgr = tf.split(rgb, 3, 3)[0]
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        input_bgr = tf.identity(rgb, name="lrp_input")

        # Architecture
        # 1. Joint regression head: res-net style structure (need to match this more closely)
        # to learn feature encoders for predicting joint positions
        # 2. Monkey masking head: conv-deconv network to mask the shape of the monkey.
        # These weights are shared with 1, but the loss is calculated seperately -- ensures
        # That we drive our joint predictions towards the monkey body.

        # 1st head:: Feature extraction and image compression
        self.conv1_1 = self.conv_layer(input_bgr, int(rgb.get_shape()[-1]), 64, "conv1_1")
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
        self.block_1 = self.resnet_block(
                in_layer=self.pool1,
                num_features=128,
                name='block_1')
        self.block_2 = self.resnet_block(
                in_layer=self.block_1,
                num_features=128,
                name='block_2')
        self.pool2 = self.max_pool(self.block_2, 'pool2')
        self.block_3 = self.resnet_block(
                in_layer=self.pool2,
                num_features=256,
                name='block_3')
        self.block_4 = self.resnet_block(
                in_layer=self.block_3,
                num_features=256,
                name='block_4')
        self.pool3 = self.max_pool(self.block_4, 'pool3')
        self.block_5 = self.resnet_block(
                in_layer=self.pool3,
                num_features=512,
                name='block_5')
        self.block_6 = self.resnet_block(
                in_layer=self.block_5,
                num_features=512,
                name='block_6')
        self.pool4 = self.max_pool(self.block_6, 'pool4')


        # Feature encoder
        resize_size = [int(x) for x in self[fe_keys[np.argmin(
            [int(self[x].get_shape()[0]) for x in fe_keys])]].get_shape()]
        new_size = np.asarray([resize_size[1], resize_size[2]])
        fe_layers = [self.batchnorm(
            tf.image.resize_bilinear(
                self[x], new_size)) for x in fe_keys]

        # 2nd head::: Masking head -- extract from fe_layers
        self.mask_head = tf.identity(self[mask_head])  # Just take the last layer assuming it's lowest res
        self.deconv1_1 = self.deconv_layer(
            self.mask_head, int(self.mask_head.get_shape()[-1]), 64, 'deconv1_1')
        self.deconv1_conv1 = self.conv_layer(
            self.deconv1_1, 64, 64, 'deconv1_conv1')
        self.deconv2_1 = self.deconv_layer(
            self.deconv1_conv1, 64, 16, 'deconv2_1')
        self.deconv2_conv1 = self.conv_layer(
            self.deconv2_1, 16, 16, 'deconv2_conv1')
        self.deconv3_1 = self.deconv_layer(
            self.deconv2_conv1, 16, 1, 'deconv3_1')

        # Combine FE heads
        self.feature_encoder = tf.concat(fe_layers, 3)
        self.feature_encoder_1x1 = self.conv_layer(
            self.feature_encoder,
            int(self.feature_encoder.get_shape()[-1]),
            64,
            "feature_encoder_1x1",
            filter_size=1)
        if train_mode is not None:
            self.feature_encoder_1x1 = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(self.feature_encoder_1x1, 0.5), lambda: self.feature_encoder_1x1)
        self.pool5 = self.max_pool(self.feature_encoder_1x1, 'pool5')
        self.fc6 = self.fc_layer(
            self.pool5,
            np.prod([int(x) for x in self.pool5.get_shape()[1:]]),
            4096,
            "fc6")
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
        self.fc8 = self.fc_layer(self.relu6, 4096, output_shape, "fc8")
        if batchnorm is not None:
            if 'fc8' in batchnorm:
                self.fc8 = self.batchnorm(self.fc8)
        final = tf.identity(self.fc8, name="lrp_output")
        self.prob = tf.nn.softmax(final, name="prob")
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



    def deconv_weights(factor, number_of_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """
        
        filter_size = get_kernel_size(factor)
        
        weights = np.zeros((filter_size,
                            filter_size,
                            number_of_classes,
                            number_of_classes), dtype=np.float32)
        
        upsample_kernel = upsample_filt(filter_size)
        
        for i in xrange(number_of_classes):
            
            weights[:, :, i, i] = upsample_kernel
        
        return weights

    def deconv_layer(
            self,
            bottom,
            in_channels,
            out_channels,
            name,
            filter_size=3,
            factor=2,
            batchnorm=None):
        with tf.variable_scope(name):
            filt = self.deconv_weights(
                factor,
                int(bottom.get_shape()[-1]))
            res = tf.nn.conv2d_transpose(
                bottom,
                filt,
                output_shape=[1, new_height, new_width, number_of_classes],
                strides=[1, factor, factor, 1])

            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)
            conv = tf.nn.transpose_conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
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

    def resnet_block(
            self,
            in_layer,
            num_features,
            name,
            num_dense_convs=2,
            num_pass_convs=0,
            dense_path_feature_prop=.25):

        shallow_path = tf.identity(in_layer)
        dense_path = tf.identity(in_layer)

        # Preallocate features
        shallow_path_features = np.repeat(num_features, num_pass_convs)
        dense_path_features = np.repeat(num_features, num_dense_convs)
        dense_path_features[:-1] = dense_path_features[:-1] * dense_path_feature_prop
        dense_path_features = np.round(dense_path_features).astype(int)

        if num_pass_convs:
            with tf.variable_scope('pass_%s' % name):
                shallow_path = self.create_path(
                    shallow_path=shallow_path,
                    num_features=num_features,
                    num_pass_convs=num_pass_convs)
                for idx, (n, f) in enumerate(
                        zip(num_pass_convs, dense_path_features)):
                    shallow_path = self.conv_layer(
                        shallow_path,
                        f,
                        f,
                        'dense_%s' % idx)
        with tf.variable_scope('dense_%s' % name):
            for idx, (n, f) in enumerate(
                    zip(num_dense_convs, shallow_path_features)):
                dense_path = self.conv_layer(
                    dense_path,
                    f,
                    f,
                    'shallow_%s' % idx)

        return shallow_path + dense_path

    def create_path(self, path, num_features, num_pass_convs):
            for idx, n in enumerate(num_pass_convs):
                shallow_path = self.conv_layer(
                    shallow_path,
                    num_features,
                    num_features,
                    'res_%s' % idx)
            return shallow_path

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

