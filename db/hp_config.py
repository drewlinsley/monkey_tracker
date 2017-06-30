import numpy as np
import itertools as it


"""
Each key in parameter_dict must be manually added to the schema.
"""
parameter_dict = {
    'lrs': np.logspace(-5, -2, 4, base=10),
    'randomize_background': np.arange(3),
    'train_batch': np.asarray([16, 32]),
    'loss_type': np.asarray(['l2', 'l1'])
}


def package_parameters():
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    return [{k: v for k, v in zip(keys_sorted, row)} for row in values]

