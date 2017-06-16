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
            target_variables=None,
            train_mode=None,
            batchnorm=None,
            fe_keys=None,
            hr_fe_keys=['pool2', 'pool3', 'pool4'],
            lr_fe_keys=['lr_pool2', 'lr_pool3']
            ):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """
        if fe_keys is not None:
            print 'You supplied feature extractor keys... These are ignored.'

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

        if 'pose' in target_variables.keys():
            if len(target_variables['pose'].get_shape()) == 1:
                pose_shape = 1
            else:
                pose_shape = int(
                    target_variables['pose'].get_shape()[-1])

        input_bgr = tf.identity(rgb, name="lrp_input")

        # Initial convolutional
        layer_structure = [
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [64, 64, None],
                'names': ['conv1_1', 'conv1_2', 'pool1'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [128, 128, None],
                'names': ['conv2_1', 'conv2_2', 'pool2'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [256, 256, None],
                'names': ['conv3_1', 'conv3_2', 'pool3'],
                'filter_size': [3, 3, None]
            }]
        output_layer = self.create_conv_tower(
            input_bgr,
            layer_structure,
            tower_name='highres_conv')
        if train_mode is not None:
            output_layer = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(output_layer, 0.5),
                lambda: output_layer)
        output_layer = tf.contrib.layers.flatten(output_layer, 'output_layer')
        self.flat_output_1 = self.fc_layer(
            output_layer,
            int(output_layer.get_shape()[-1]),
            4096,
            'flat_output_1')
        if train_mode is not None:
            output_layer = tf.cond(
                train_mode,
                lambda: tf.nn.dropout(output_layer, 0.5),
                lambda: output_layer)
        self.flat_output_2 = self.fc_layer(
            self.flat_output_1,
            int(self.flat_output_1.get_shape()[-1]),
            4096,
            'flat_output_2')

        if 'label' in target_variables.keys():
            self.output = self.fc_layer(
                self.flat_output_2,
                int(self.flat_output_2.get_shape()[-1]),
                output_shape,
                "output")

        if 'occlusion' in target_variables.keys():
            # Occlusion head
            self.occlusion = self.fc_layer(
                self.flat_output_2,
                int(self.flat_output_2.get_shape()[-1]),
                occlusion_shape,
                "occlusion")
        self.data_dict = None

        if 'pose' in target_variables.keys():
            # Occlusion head
            self.pose = self.fc_layer(
                self.high_1x1_2_pool,
                int(self.high_1x1_2_pool.get_shape()[-1]),
                pose_shape,
                "pose")

        # Deconv to full image
        layer_structure = [
            {
                'layers': ['conv', 'deconv'],
                'weights': [256, 256],
                'names': ['up-conv1_1', 'up-conv1_2'],
                'filter_size': [3, 3]
            },
            {
                'layers': ['conv', 'deconv'],
                'weights': [64, 64],
                'names': ['up-conv2_1', 'up-conv2_2'],
                'filter_size': [3, 3]
            },
            {
                'layers': ['conv', 'deconv'],
                'weights': [1, 1],
                'names': ['up-conv3_1', 'up-conv3_2'],
                'filter_size': [3, 3]
            }
            ]
        self.deconv = self.create_conv_tower(
            input_bgr,
            layer_structure,
            tower_name='highres_conv')

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
                        act, argmax_act = self.max_pool(
                            bottom=act,
                            name=na)
                        setattr(self, na + '_argmax', argmax_act)
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
                            layer_name=na
                    )
                    elif la == 'deconv':
                        act = self.deconv_layer(
                            bottom=act,
                            in_channels=int(act.get_shape()[-1]),
                            out_channels=we,
                            layer_name=na,
                            filter_size=fs
                    )
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
        return tf.nn.max_pool_with_argmax(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(
            self,
            bottom,
            in_channels,
            out_channels,
            name,
            filter_size=3,
            batchnorm=None,
            stride=[1, 1, 1, 1]):
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

    def deconv_layer(
            self,
            bottom,
            in_channels,
            out_channels,
            filter_size,
            layer_name,
            stride=[1, 1, 1, 1],
            padding='SAME'):
        conv_name = layer_name.split('-')[-1]
        # argmax_name = conv_name + '_argmax'
        with tf.variable_scope(layer_name):
            act = self.deconv(
                bottom,
                filter_size,
                in_channels,
                out_channels,
                layer_name)
            return self.unpool_layer2x2(
                act,
                self[conv_name],
                tf.shape(self[conv_name]))

    def deconv(
            self,
            bottom,
            filter_size,
            in_channels,
            out_channels,
            name,
            strides=[1, 1, 1, 1],
            padding='SAME'):
        W, b = self.get_conv_var(
            filter_size,
            in_channels,
            out_channels,
            name)

        bottom_shape = tf.shape(bottom)
        out_shape = tf.stack(
            [bottom_shape[0], bottom_shape[1], bottom_shape[2], out_channels])

        return tf.nn.conv2d_transpose(
            bottom,
            W,
            out_shape,
            strides=strides,
            padding=padding) + b

    def unpool_layer2x2(self, x, raveled_argmax, out_shape):
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(
            t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(
            tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(
            self,
            filter_size,
            in_channels,
            out_channels,
            name,
            init_type='xavier'):
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
