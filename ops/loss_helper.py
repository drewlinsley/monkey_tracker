import tensorflow as tf


def potential_aux_losses():
 return [
    {'occlusion': {
        'y_name': 'occlusion',
        'model_name': 'occlusion',
        'loss_function': 'sigmoid',
        'var_label': 'occlusionhead'
        }
    },
    {'z': {
        'y_name': 'z',
        'model_name': 'z',
        'loss_function': 'l2',
        'var_label': 'z head'
        }
    },
    {'size': {
        'y_name': 'size',
        'model_name': 'size',
        'loss_function': 'l2',
        'var_label': 'size head'
        }
    },
    {'pose': {
        'y_name': 'pose',
        'model_name': 'pose',
        'loss_function': 'l2',
        'var_label': 'pose head'
        }
    },
    {'deconv': {
        'y_name': 'image',
        'model_name': 'deconv',
        'loss_function': 'l2',
        'var_label': 'deconv head'
        }
    },
    {'im_label': {
        'y_name': 'im_label',
        'model_name': 'rgb',
        'loss_function': 'l2',
        'var_label': 'deconv head'
        }
    }
    ]

def get_aux_losses(
        loss_list,
        loss_label,
        train_data_dict,
        model,
        aux_loss_dict):
    aux_dict = aux_loss_dict.values()[0]
    if aux_loss_dict.keys()[0] in train_data_dict.keys():
        y = train_data_dict[aux_dict['y_name']]
        yhat = model[aux_dict['model_name']]
        output_label = aux_dict['var_label']
        loss_function = aux_dict['loss_function']
        if loss_function == 'sigmoid':
            loss_list += [tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y,
                    logits=yhat))]
        elif loss_function == 'l2':
            loss_list += [tf.nn.l2_loss(
                y - yhat)]
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
    else:
        raise 'Unidentified optimizer'
    return optimizer


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

