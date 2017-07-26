import tensorflow as tf
import yellowfin
from tensorflow.python.framework import ops


def potential_aux_losses():
    return [
        {
            'occlusion': {
                'y_name': 'occlusion',
                'model_name': 'occlusion',
                'loss_function': 'sigmoid',
                'var_label': 'occlusionhead',
                'lambda': 0.1,
                'aux_fun': 'none',
                'da_override': True
                }
        },
        {
            'z': {
                'y_name': 'z',
                'model_name': 'z',
                'loss_function': 'l2',
                'var_label': 'z head',
                'lambda': 0.01,
                'aux_fun': 'none',
                'da_override': True
                }
        },
        {
            'size': {
                'y_name': 'size',
                'model_name': 'size',
                'loss_function': 'l2',
                'var_label': 'size head',
                'lambda': 0.1,
                'aux_fun': 'none',
                'da_override': False
                }
        },
        {
            'pose': {
                'y_name': 'pose',
                'model_name': 'pose',
                'loss_function': 'l2',
                'var_label': 'pose head',
                'lambda': 0.1,
                'aux_fun': 'none',
                'da_override': True
                }
        },
        {
            'deconv': {
                'y_name': 'image',
                'model_name': 'deconv',
                'loss_function': 'l2',
                'var_label': 'deconv head',
                'lambda': None,
                'aux_fun': 'resize',
                'da_override': True
                }
        },
        {
            'deconv_label': {
                'y_name': 'deconv_label',
                'model_name': 'deconv',
                'loss_function': 'cce',
                'var_label': 'deconv head',
                'lambda': {14: 0.01},  # reduce weight on background
                'aux_fun': 'resize',
                'da_override': True
                }
        },
        {
            'domain_adaptation': {
                'y_name': 'domain_adaptation',
                'model_name': 'domain_adaptation',
                'loss_function': 'cce',
                'var_label': 'domain head',
                'lambda': 0.1,
                'aux_fun': 'none',
                'da_override': False
                }
        }
        ]


def get_aux_losses(
        loss_list,
        loss_label,
        train_data_dict,
        model,
        aux_loss_dict,
        domain_adaptation=False):
    aux_dict = aux_loss_dict.values()[0]
    if aux_loss_dict.keys()[0] in train_data_dict.keys():
        output_label = aux_dict['var_label']
        if domain_adaptation and aux_dict['da_override']:
            loss = 0.
        else:
            y = train_data_dict[aux_dict['y_name']]
            yhat = model[aux_dict['model_name']]
            loss_function = aux_dict['loss_function']
            reg_weight = aux_dict['lambda']
            aux_fun = aux_dict['aux_fun']
            if aux_fun == 'resize':
                y = tf.image.resize_bilinear(
                    y, [int(x) for x in yhat.get_shape()[1:3]])
            if loss_function == 'sigmoid':
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=y,
                        logits=yhat))
            elif loss_function == 'cce':
                if isinstance(reg_weight, dict):
                    # Only using index 1 for now
                    index = reg_weight.keys()[0]
                    weight = reg_weight.values()[0]
                    split_tensor = tf.split(
                        y,
                        int(y.get_shape()[-1]), axis=3)[index]
                    weights = split_tensor * weight  # Weight for bg loss
                    weights = tf.squeeze(weights + tf.cast(tf.equal(
                        weights, 0), tf.float32))  # No weight on fg
                    inter_loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=yhat,
                        labels=y,
                        dim=-1)
                    loss = tf.reduce_mean(inter_loss * weights)
                else:
                    loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.cast(tf.squeeze(y), tf.int32),
                            logits=yhat))
            elif loss_function == 'l2':
                loss = tf.nn.l2_loss(y - yhat)
            if reg_weight is not None and not isinstance(reg_weight, dict):
                loss *= reg_weight
        loss_list += [loss]
        loss_label += [output_label]
    return loss_list, loss_label


def return_optimizer(optimizer):
    if optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer
    elif optimizer == 'momentum':
        #  momentum_var = tf.placeholder(tf.float32, shape=(1))
        optimizer = lambda x: tf.train.MomentumOptimizer(x, momentum=0.1)
    elif optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer
    elif optimizer == 'yellowfin':
        optimizer = yellowfin.YFOptimizer
    else:
        raise RuntimeError('Unidentified optimizer')
    return optimizer


class FlipGrad(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
        self.num_calls += 1
        return y


flipped_gradient = FlipGrad()

