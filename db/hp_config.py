import numpy as np
import itertools as it


"""
Each key in parameter_dict must be manually added to the schema.
"""
parameter_dict = {
    'lr': np.asarray([1e-5]),  # np.logspace(-5, -2, 4, base=10),
    'randomize_background': np.asarray([None, 2]),  # np.arange(3),
    'aux_losses': np.asarray([[None], ['size', 'occlusion']]),
    'model_type': np.asarray(['cnn_multiscale_high_res_low_res_skinny_pose_occlusion.py'], ['small_cnn_multiscale_high_res_low_res_skinny_pose_occlusion_bigger_lr'], ['cnn_multiscale_high_res_atrous_skinny_pose_occlusion'])
}


# def package_parameters():
#     keys_sorted = sorted(parameter_dict)
#     values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
#     return [{k: v for k, v in zip(keys_sorted, row)} for row in values]


def package_parameters():
    """
    Each key in parameter_dict must be manually added to the schema.
    """
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    combos = tuple({k: v for k, v in zip(keys_sorted, row)} for row in values)
    # Really dumb but whatever
    print 'Derived %s combinations.' % len(combos)
    return combos
