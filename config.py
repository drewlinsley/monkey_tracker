import os
from ops import babas_files_list

class monkeyConfig(object):
    def __init__(self): 
        '''Main config file for monkey tracking'''

        # Directory settings
        self.base_dir = '/media/data_cifs/monkey_tracking/' #'/media/data_cifs/monkey_tracking/batches/MovieRender'
        self.results_dir = os.path.join(self.base_dir, 'results', 'thisisatest') #'/media/data_cifs/monkey_tracking/batches/MovieRender'
        self.image_dir = os.path.join(self.base_dir, 'batches', 'thisisatest') #os.path.join(self.base_dir, 'walk-all-png') 
        self.depth_dir = os.path.join(self.image_dir, 'depth', 'true_depth')
        self.label_dir = os.path.join(self.image_dir, 'labels', 'joint_coords')
        self.npy_dir = os.path.join(self.base_dir, 'batches', 'test')  # Output for numpys
        self.pixel_label_dir = os.path.join(self.image_dir, 'labels', 'pixel_joint_coords')  # 'joint_coords')
        self.occlusion_dir = os.path.join(self.image_dir, 'labels', 'occlusions')  # Set to None if there's no occlusion data
        self.im_label_dir = os.path.join(self.image_dir, 'labels', 'npyLabels')
        self.depth_regex = '*[0-9].npy'
        self.image_extension = '.npy'
        self.label_extension = '.npy' 
        self.occlusion_extension = '.npy'
        self.model_output = os.path.join(self.results_dir, 'model_output') 
        self.tfrecord_dir = os.path.join(self.image_dir, 'tfrecords_fast')
        self.train_summaries = os.path.join(self.results_dir, 'summaries')
        self.train_checkpoint = os.path.join(self.results_dir, 'checkpoints')
        self.weight_npy_path = None  # os.path.join('/media/data_cifs/monkey_tracking/saved_weights/cnn_multiscale_high_res_low_res_skinny_pose_occlusion.npy')
        use_checkpoint = False
        if use_checkpoint:
            self.model_name = 'skip_res_small_conv_deconv_2017_08_30_12_06_56'
            # self.model_name = 'small_cnn_multiscale_high_res_low_res_skinny_pose_occlusion_bigger_lr_2017_09_11_17_30_00'
            self.ckpt_file = None
            self.resume_from_checkpoint = os.path.join(
                self.model_output,
                self.model_name)
            if 'ckpt' in self.resume_from_checkpoint.split('/')[-1]:
                self.saved_config = '%s.npy' % os.path.sep.join(self.resume_from_checkpoint.split('/')[:-1])
            else:
                self.saved_config = '%s.npy' % self.resume_from_checkpoint
            self.segmentation_model_name = 'small_cnn_multiscale_high_res_low_res_skinny_pose_occlusion_bigger_lr_reduced_2017_08_14_19_28_13'
            self.segmentation_resume_from_checkpoint = os.path.join(
                self.model_output,
                self.segmentation_model_name)
            self.segmentation_saved_config = '%s.npy' % self.resume_from_checkpoint
        else:
            self.resume_from_checkpoint = None

        # Tfrecords
        self.use_train_as_val = False
        self.new_tf_names = {'train': 'train.tfrecords', 'val': 'val.tfrecords'}  # {'train': 'train_2mill.tfrecords', 'val': 'val_2mill.tfrecords'}
        self.train_tfrecords = 'train.tfrecords'  # Decouple the these vars so you can create new records while training #'train_2mill.tfrecords' 
        self.val_tfrecords = 'val.tfrecords'  # 'val_2mill.tfrecords'        
        self.max_train = None  # Limit the number of files we're going to store in a tfrecords. Set to None if there's no limit.
        self.max_depth = 1200.  # Maya: 1200. --note: prepare kinect with lower value than during testing (e.g. 900train/1800test). Divide each image by this value to normalize it to [0, 1]. This is the only normalization we will do. Must be a float!
        self.min_depth = 200.  # Maya: 200. Use for normalizing the kinect data
        self.background_constant = self.max_depth * 2  # HIGH_NUMBER
        self.resize = [240, 320, 3]  # CNN input (don't change) -- make sure this the same dimensions as the input
        self.image_input_size = [480, 640]  # Maya render output
        self.image_target_size = [240, 320, 3]  # Resize before tfrecords
        self.image_target_size_is_flipped = True  # A flip was introduced for labels: h/w -> x/y.
        self.maya_conversion = 640.0 / 500.0  # pixels / maya units
        self.sample = {'train': True, 'val': False}  # random sample of feats
        self.use_image_labels = False  # if true, extract  color-labeled images
        self.use_pixel_xy = True
        self.background_multiplier = 1.01  # Where to place the imaginary wall in the renders w.r.t. the max depth value
        self.randomize_background = 1.
        self.augment_background = 'background_perlin'  #  'background'  # 'background_perlin'  # 'background_perlin'  # 'background_perlin'  # 'perlin'  # 'rescale' 'perlin' 'constant' 'rescale_and_perlin'
        self.background_folder = 'backgrounds'

        # Model settings
        self.epochs = 1000
        # self.model_type = 'skip_small_conv_deconv'
        self.model_type = 'small_cnn_multiscale_high_res_low_res_skinny_pose_occlusion'
        # self.model_type = 'small_cnn_multiscale_high_res_low_res_skinny_pose_occlusion_bigger_lr_reduced'
        # self.model_type = 'small_cnn_multiscale_high_res_low_res_skinny_pose_occlusion_bigger_lr'
        self.fine_tune_layers = None
        self.batch_norm = [None]  # ['high_feature_encoder_1x1_0', 'high_feature_encoder_1x1_1', 'high_feature_encoder_1x1_2']  # ['fc6', 'fc7', 'pre_fc8']
        self.data_augmentations = [
            'left_right',
            # 'up_down'
        ]
        self.convert_labels_to_pixel_space = True

        # Key training settings
        self.train_batch = 16
        self.validation_batch = 16
        self.ratio = None  # [0.1, 0.9]
        self.lr = 3e-4  # Tune this -- also try SGD instead of ADAm
        self.hold_lr = 1e-8
        self.keep_checkpoints = 100
        self.optimizer = 'adam'
        self.steps_before_validation = 1000
        self.loss_type = 'l1'
        self.grad_clip = False
        self.dim_weight = [1, 1, 0.25]   # If including xyz dim-specific weighting

        # Potentially outdated training settings
        self.use_training_loss = False  # early stopping based on loss
        self.early_stopping_rounds = 100
        self.test_proportion = 0.1  # TEST_RATIO
        self.mean_file = 'mean_file'  # Double check: used in training?

        # Auxillary training settings
        self.normalize_labels = True
        self.aux_losses = ['occlusion', 'stretch_renders']  # , 'deconv_label']  # ['z', 'size', 'occlusion', 'deconv_label']  # 'occlusion' 'pose' 'size' 'z' 'deconv_label' 'deconv'
        self.calculate_per_joint_loss = False  # BABAS 'skeleton and joint'  # False
        self.include_validation = True  # BABAS '/media/data_cifs/monkey_tracking/data_for_babas/11_8_17_out_of_bag_val_1/val.tfrecords'  # '/media/data_cifs/monkey_tracking/data_for_babas/10_10_17_out_of_bag_val/val.tfrecords'
        self.wd_type = 'l2'
        self.wd_penalty = 5e-7
        self.wd_layers = ['high_feature_encoder_1x1_0', 'high_feature_encoder_1x1_1', 'high_feature_encoder_1x1_2']  # ['fc6', 'fc7', 'pre_fc8']
        self.fc_lambda = 0.01

        # Labels for the rendered images
        self.labels = {
            'back_torso':      (99,  130,   0, 254),
            'background':      (0,     0,   0,   0),
            'front_torso':     (250, 200, 189, 254),
            'head':            (199,   0,   0, 254),
            'hip':             (130,  50, 120, 254),
            'left_finger':     (200,   0, 200, 254),
            'left_bart':       (150, 130, 139, 254),
            'left_hand':       (250,   0, 250, 254),
            'left_lower_arm':  (0,   250,   0, 254),
            'left_lower_leg':  (0,     0, 250, 254),
            'left_shoulder':   (50,   90, 139, 254),
            'left_toe':        (170, 190,  39, 254),
            'left_upper_arm':  (249, 180,   0, 254),
            'left_upper_leg':  (240, 240, 239, 254),
            'neck':            (70,   70,  69, 254),
            'right_finger':    (249,  50,   0, 254),
            'right_bart':      (0,   150,   0, 254),
            'right_hand':      (249, 250,   0, 254),
            'right_lower_arm': (249,   0,   0, 254),
            'right_lower_leg': (249, 100,   0, 254),
            'right_shoulder':  (170,   0, 250, 254),
            'right_toe':       (100, 150, 100, 254),
            'right_upper_arm': (0,   250, 249, 254),
            'right_upper_leg': (20,   20,  19, 254),
            'tail':            (100, 180, 249, 254)
        }

        self.joint_order = [
            'lEye',
            'neck', 
            'abdomen', 
            'lShldr', 
            'rShldr', 
            'lForeArm', 
            'rForeArm', 
            'lHand', 
            'rHand', 
            'lMid1', 
            'rMid1', 
            'lMid3', 
            'rMid3', 
            'lThigh', 
            'rThigh',
            'lShin', 
            'rShin', 
            'lFoot', 
            'rFoot', 
            'lToe', 
            'rToe', 
            'lToeMid3', 
            'rToeMid3'
        ]

        self.joint_names = [
            'head',
            'neck',
            'abdomen',
            'left shoulder',
            'right shoulder',
            'left elbow',
            'right elbow',
            'left hand',
            'right hand',
            'left knuckles',
            'right knuckles',
            'left fingertips',
            'right fingertips',
            'left hip',
            'right hip',
            'left knee',
            'right knee',
            'left ankle',
            'right ankle',
            'left bart',
            'right bart',
            'left toetips',
            'right toetips'
        ]

        self.joint_graph = {
            'head': 'neck',
            'neck': 'abdomen',
            'abdomen': 'neck',
            'left shoulder': 'neck',
            'right shoulder': 'neck',
            'left elbow': 'left shoulder',
            'right elbow': 'right shoulder',
            'left hand': 'left elbow',
            'right hand': 'right elbow',
            'left knuckles': 'left hand',
            'right knuckles': 'right hand',
            'left fingertips': 'left knuckles',
            'right fingertips': 'right knuckles',
            'left hip': 'abdomen',
            'right hip': 'abdomen',
            'left knee': 'left hip',
            'right knee': 'right hip',
            'left ankle': 'left knee',
            'right ankle': 'right knee',
            'left bart': 'left ankle',
            'right bart': 'right ankle',
            'left toetips': 'left bart',
            'right toetips': 'right bart'
        }


        self.selected_joints = None  # ['lThigh', 'lShin', 'lFoot', 'lToe', 'lToeMid3']  #  None  # ['lEye']  # Set to None to ignore
        self.num_dims = 3
        self.keep_dims = 3
        self.num_classes = len(self.joint_order) * self.num_dims
        self.mask_occluded_joints = False
        self.babas_file_for_import = babas_files_list.data()
        self.babas_tfrecord_dir = None  # BABAS '/media/data_cifs/monkey_tracking/data_for_babas/11_8_17_out_of_bag_val_1'  # '/media/data_cifs/monkey_tracking/batches/TrueDepth2MilStore/tfrecords_fast/val.tfrecords'  # True

        # Feature extraction settings for classic kinect alg
        self.offset_nn = 30  # random +/- x,y pixel offset range # Tune this
        self.n_features = 400  # Tune this
        self.max_pixels_per_image = 800  # Tune this
        self.cte_depth = 2  # ??

