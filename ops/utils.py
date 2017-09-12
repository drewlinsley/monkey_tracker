"""Basic python utilities."""

import os
import re
import numpy as np
from glob import glob
from datetime import datetime


def get_files(directory, pattern):
    return np.asarray(sorted(glob(os.path.join(directory, pattern))))


def get_dt():
    return re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')


def import_cnn(model_type, model_dir='models'):
    return getattr(
        __import__(model_dir, fromlist=[model_type]), model_type)


def save_training_data(output_dir, data, name):
    np.save(
        os.path.join(
            output_dir,
            name),
        data)
