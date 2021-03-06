#Embedded file name: /home/dmurphy/monkey_tracker/ops/data_processing_joints.py
import os
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ops.utils import get_files, get_files_fast
from scipy import misc
from skimage.transform import resize
import time
from multiprocessing import Pool, Process, Manager

def rescale_zo(image):
    mi = np.min(image, keepdims=True).astype(np.float)
    ma = np.max(image, keepdims=True).astype(np.float)
    return (image - mi) / (ma - mi)


def bytes_feature(values):
    """Encodes an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Encodes an int list into a tf int64 list for a tfrecord."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def encode_example(im, label, occlusion = None, im_label = None):
    """Encodes a single label/image/feature example into a tfrecord entry."""
    feature = {'label': bytes_feature(label.tostring()),
     'image': bytes_feature(im.tostring())}
    if im_label is not None:
        feature['im_label'] = bytes_feature(im_label.tostring())
    if occlusion is not None:
        feature['occlusion'] = bytes_feature(occlusion.tostring())
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def cv_files(df, lf, of = None, pf = None, val_prop = 0.1):
    """Creates training/validation indices and outputs a dict with
    split depth/label files."""
    rand_idx = np.random.permutation(np.arange(len(df)))
    train_cut = int((1 - val_prop) * len(df))
    train_idx = rand_idx[:train_cut]
    val_idx = rand_idx[train_cut:]
    depth_files = {'train': df[train_idx],
     'val': df[val_idx]}
    label_files = {'train': lf[train_idx],
     'val': lf[val_idx]}
    if of is not None:
        occlusion_files = {'train': of[train_idx],
         'val': of[val_idx]}
    else:
        occlusion_files = {'train': None,
         'val': None}
    if pf is not None:
        pixel_label_files = {'train': pf[train_idx],
         'val': pf[val_idx]}
    else:
        pixel_label_files = {'train': None,
         'val': None}
    return (depth_files,
     label_files,
     occlusion_files,
     pixel_label_files)


def create_joint_tf_records(depth_files, label_files, pixel_label_files, tf_file, config, occlusions = None, im_label = None):
    """Feature extracts and creates the tfrecords."""
    im_list, num_successful = [], 0
    if depth_files[0].split('.')[1] == 'npy':
        use_npy = True
        print('Getting depth from npys.')
    else:
        use_npy = False
        print('Getting depth from images.')
    if config.max_train is not None:
        num_files = config.max_train
    else:
        num_files = len(depth_files)
        tf_dir = '/'.join(tf_file.split('/')[:-1])
    if not os.path.exists(tf_dir):
        os.makedirs(tf_dir)
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i, (depth, label) in tqdm(enumerate(zip(depth_files, label_files)), total=num_files):
            if use_npy:
                depth_image = np.load(depth)[:, :, :3].astype(np.float32)
            else:
                depth_image = misc.imread(depth, mode='F')[:, :, :3]
            depth_image[depth_image == depth_image.min()] = 0.0
            depth_image[np.isnan(depth_image)] = 0.0
            if depth_image.sum() > 0:
                if config.image_input_size != config.image_target_size:
                    depth_image = resize(depth_image, config.image_target_size[:2], preserve_range=True, order=0)
                label_vector = np.load(label).astype(np.float32)
                if config.use_pixel_xy:
                    pixel_label_vector = np.load(pixel_label_files[i]).astype(np.float32)
                    label_vector = pixel_label_vector
                if config.use_image_labels:
                    im_label = np.load(os.path.join(config.im_label_dir, re.split(config.label_extension, re.split('/', label)[-1])[0] + config.image_extension))[:, :, :3]
                    im_label[np.isnan(im_label)] = 0
                    if config.image_input_size != config.image_target_size:
                        im_label = misc.imresize(im_label, config.image_target_size[:2])
                    im_label = im_label.astype(np.float32)
                im_list.append(np.mean(depth_image))
                if occlusions is not None:
                    occlusion = np.load(occlusions[i]).astype(np.float32)
                else:
                    occlusion = None
                example = encode_example(im=depth_image.astype(np.float32), label=label_vector, im_label=im_label, occlusion=occlusion)
                tfrecord_writer.write(example)
                num_successful += 1
            if config.max_train is not None and num_successful > config.max_train:
                break

    return im_list


def extract_depth_features_into_tfrecord(depth_files, label_files, pixel_label_files, occlusion_files, config):
    """Prepares the tf op for nearest neighbor depth features and runs op to
    make tfrecords from them."""
    depth_files, label_files, occlusion_files, pixel_label_files = cv_files(depth_files, label_files, occlusion_files, pixel_label_files)
    mean_dict = {}
    for k in list(depth_files.keys()):
        print(('Getting depth features: %s' % k))
        im_means = create_joint_tf_records(depth_files=depth_files[k], label_files=label_files[k], pixel_label_files=pixel_label_files[k], tf_file=os.path.join(config.tfrecord_dir, config.new_tf_names[k]), config=config, occlusions=occlusion_files[k])
        mean_dict[k + '_image'] = im_means

    print('Finished')
    return mean_dict


def process_data(config):
    """Extract nearest neighbor features from depth images.
    Eventually move this into the tensorflow graph and handle
    depth files/ label files in batches. Also directly add
    images into a tfrecord instead of into a numpy array."""
    depth_files = get_files(config.depth_dir, config.depth_regex)
    label_files = np.asarray([ os.path.join(config.label_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.label_extension) for x in depth_files ])
    pixel_label_files = np.asarray([ os.path.join(config.pixel_label_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.label_extension) for x in depth_files ])
    occlusion_files = np.asarray([ os.path.join(config.occlusion_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.occlusion_extension) for x in depth_files ])
    if not os.path.isfile(occlusion_files[0]):
        occlusion_files = None
    mean_dict = extract_depth_features_into_tfrecord(depth_files=depth_files, label_files=label_files, pixel_label_files=pixel_label_files, occlusion_files=occlusion_files, config=config)
    np.savez(os.path.join(config.tfrecord_dir, config.mean_file), data=mean_dict, **mean_dict)


def write_files(q, total_files, tf_file):
    with tf.python_io.TFRecordWriter(tf_file) as tfrecord_writer:
        for i in tqdm(list(range(total_files))):
            if q.empty():
                print('waiting for more files to encode')
            depth_image, label_vector, im_label, occlusion = q.get()
            example = encode_example(im=depth_image.astype(np.float32), label=label_vector, im_label=im_label, occlusion=occlusion)
            tfrecord_writer.write(example)


def adjust_files(array):
    q, depth_file, label_file, pixel_label_file, occlusion_file, config = array
    while q.qsize() > 100:
        print('waiting for more files to be encoded before loading more')
        sleep(1)

    depth_image = np.load(depth_file)[:, :, :3].astype(np.float32)
    depth_image[depth_image == depth_image.min()] = 0.0
    depth_image[np.isnan(depth_image)] = 0.0
    if depth_image.sum() > 0:
        if config.image_input_size != config.image_target_size:
            depth_image = resize(depth_image, config.image_target_size[:2], preserve_range=True, order=0)
        if config.use_pixel_xy:
            pixel_label_vector = np.load(pixel_label_file).astype(np.float32)
            label_vector = pixel_label_vector
        else:
            label_vector = np.load(label_file).astype(np.float32)
        if config.use_image_labels:
            im_label = np.load(os.path.join(config.im_label_dir, re.split(config.label_extension, re.split('/', label_file)[-1])[0] + config.image_extension))[:, :, :3]
            im_label[np.isnan(im_label)] = 0
            if config.image_input_size != config.image_target_size:
                im_label = resize(im_label, config.image_target_size[:2])
            im_label = im_label.astype(np.float32)
        else:
            im_label = None
        if occlusion_file is not None:
            occlusion = np.load(occlusion_file).astype(np.float32)
        else:
            occlusion = None
        q.put((depth_image.astype(np.float32),
         label_vector,
         im_label,
         occlusion))


def process_data_fast(config):
    print('using fast version')
    if not os.path.exists(config.tfrecord_dir):
        os.makedirs(config.tfrecord_dir)
    depth_files = get_files_fast(config.depth_dir, config.depth_regex)
    label_files = np.asarray([ os.path.join(config.label_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.label_extension) for x in depth_files ])
    pixel_label_files = np.asarray([ os.path.join(config.pixel_label_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.label_extension) for x in depth_files ])
    if config.occlusion_dir is not None:
        occlusion_files = np.asarray([ os.path.join(config.occlusion_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.occlusion_extension) for x in depth_files ])
    else:
        occlusion_files = None
    depth_files, label_files, occlusion_files, pixel_label_files = cv_files(depth_files, label_files, occlusion_files, pixel_label_files)
    mean_dict = {}
    processes = []
    m = Manager()
    for k in list(depth_files.keys()):
        print(('Getting depth features: %s' % k))
        q = m.Queue()
        tf_file = os.path.join(config.tfrecord_dir, config.new_tf_names[k])
        pool = Pool(20)
        pool.imap_unordered(adjust_files, [ [q,
         d,
         l,
         p,
         o,
         config] for d, l, p, o in zip(depth_files[k], label_files[k], pixel_label_files[k], occlusion_files[k]) ])
        p = Process(target=write_files, args=(q, len(depth_files[k]), tf_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def process_data_monitor(config, monitor_file, total_files):
    """Extract nearest neighbor features from depth images.
    Eventually move this into the tensorflow graph and handle
    depth files/ label files in batches. Also directly add
    images into a tfrecord instead of into a numpy array."""
    f = open(monitor_file, 'r')
    num_processed = 0
    depth_files = []
    mean_dict = {}
    writers = {}
    for k in ['train', 'val']:
        tf_file = os.path.join(config.tfrecord_dir, config.new_tf_names[k])
        writers[k] = tf.python_io.TFRecordWriter(tf_file)

    while num_processed < total_files:
        position = f.tell()
        depth_file = f.readline()
        if not depth_file:
            time.sleep(1)
            continue
        if depth_file[-1] != '\n':
            time.sleep(1)
            f.seek(position)
            continue
        depth_file = depth_file[:-1]
        depth_files.append(depth_file)
        num_processed += 1
        if len(depth_files) == 1000:
            print(('%s out of %s files have been processed' % (num_processed - 1000, total_files)))
            depth_files = np.asarray(depth_files)
            label_files = np.asarray([ os.path.join(config.label_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.label_extension) for x in depth_files ])
            pixel_label_files = np.asarray([ os.path.join(config.pixel_label_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.label_extension) for x in depth_files ])
            occlusion_files = np.asarray([ os.path.join(config.occlusion_dir, re.split(config.image_extension, re.split('/', x)[-1])[0] + config.occlusion_extension) for x in depth_files ])
            if not os.path.isfile(occlusion_files[0]):
                occlusion_files = None
            depth_files, label_files, occlusion_files, pixel_label_files = cv_files(depth_files, label_files, occlusion_files, pixel_label_files)
            for k in list(depth_files.keys()):
                print(('Getting depth features: %s' % k))
                im_list = []
                if depth_files[k][0].split('.')[1] == 'npy':
                    use_npy = True
                    print('Getting depth from npys.')
                else:
                    use_npy = False
                    print('Getting depth from images.')
                num_files = len(depth_files[k])
                tf_dir = config.tfrecord_dir
                if not os.path.exists(tf_dir):
                    os.makedirs(tf_dir)
                for i, (depth, label) in tqdm(enumerate(zip(depth_files[k], label_files[k])), total=num_files):
                    if use_npy:
                        depth_image = np.load(depth)[:, :, :3].astype(np.float32)
                    else:
                        depth_image = misc.imread(depth, mode='F')[:, :, :3]
                    depth_image[depth_image == depth_image.min()] = 0.0
                    depth_image[np.isnan(depth_image)] = 0.0
                    if depth_image.sum() > 0:
                        if config.image_input_size != config.image_target_size:
                            depth_image = resize(depth_image, config.image_target_size[:2], preserve_range=True, order=0)
                        label_vector = np.load(label).astype(np.float32)
                        if config.use_pixel_xy:
                            pixel_label_vector = np.load(pixel_label_files[k][i]).astype(np.float32)
                            label_vector = pixel_label_vector
                        if config.use_image_labels:
                            im_label = np.load(os.path.join(config.im_label_dir, re.split(config.label_extension, re.split('/', label)[-1])[0] + config.image_extension))[:, :, :3]
                            im_label[np.isnan(im_label)] = 0
                            if config.image_input_size != config.image_target_size:
                                im_label = resize(im_label, config.image_target_size[:2])
                            im_label = im_label.astype(np.float32)
                        im_list.append(np.mean(depth_image))
                        if occlusion_files is not None:
                            occlusion = np.load(occlusion_files[k][i]).astype(np.float32)
                        else:
                            occlusion = None
                        example = encode_example(im=depth_image.astype(np.float32), label=label_vector, im_label=im_label, occlusion=occlusion)
                        writers[k].write(example)

                if k + '_image' not in mean_dict:
                    mean_dict[k + '_image'] = []
                mean_dict[k + '_image'] += im_list

            depth_files = []

    np.savez(os.path.join(config.tfrecord_dir, config.mean_file), data=mean_dict, **mean_dict)
