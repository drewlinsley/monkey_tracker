import os
from ops import babas_files_list


class monkeyConfig(object):
    def __init__(self):
        '''Main config file for monkey tracking'''

        # Directory settings
        self.base_dir = '/media/data_cifs/monkey_tracking/'

        # Data directories
        self.render_version = ''  # TODO: Move all renders into a v1 dir
        self.depth_pattern = 'depth_*.png'
        self.label_label = 'joints'
        self.label_ext = 'txt'
        self.depth_label = 'depth'
        self.occlusion_label = None
        self.occlusion_ext = None
        self.part_label = None
        self.part_ext = None
        self.render_directory = os.path.join(
            self.base_dir,
            'monkey_renders',
            self.render_version)

        # TF records
        self.tfrecord_id = 'lakshmi_first_pass'
        self.tfrecord_dir = os.path.join(
            self.base_dir,
            'tfrecords')
        self.use_train_as_val = False
        self.train_tfrecords = '%s_train.tfrecords' % self.tfrecord_id
        self.val_tfrecords = '%s_val.tfrecords' % self.tfrecord_id
        self.cv_type = None  # only_train_data, leave_movie_out, None

        # Training output
        self.results_dir = os.path.join(
            self.base_dir,
            'results')
        self.train_summaries = os.path.join(
            self.base_dir,
            'summaries')
        self.train_checkpoint = os.path.join(
            self.base_dir,
            'checkpoints')

        # Data loading settings
        self.max_train_files = None  # Limit the number of files we're going to store in a tfrecords. Set to None if there's no limit.
        self.max_depth = 10000.
        self.min_depth = 0.
        self.mixing_dict = {
            0: 1,
            1: 3
        }
        self.background_constant = 10000.
        self.image_target_size = [424, 512, 1]  # Resize before tfrecords
        self.resize = [424, 512, 1]  # CNN input (don't change) -- make sure this the same dimensions as the input
        self.include_validation = '/media/data_cifs/monkey_tracking/data_for_babas/2_8_17_out_of_bag_val/lakshmi_first_pass_val.tfrecords'  # Validate on babas data
        self.babas_file_for_import = babas_files_list.data()
        self.babas_tfrecord_dir = '/media/data_cifs/monkey_tracking/data_for_babas/2_8_17_out_of_bag_val'  # '/media/data_cifs/monkey_tracking/batches/TrueDepth2MilStore/tfrecords_fast/val.tfrecords'  # True

        # Model initialization settings
        self.model_weight_path = {
            'vgg16': os.path.join(
                '%smedia' % os.path.sep,
                'data_cifs',
                'clicktionary',
                'pretrained_weights',
                'vgg16.npy'),
            'inceptionv3': os.path.join(
                '%smedia' % os.path.sep,
                'data_cifs',
                'clicktionary',
                'pretrained_weights',
                'inceptionv3.npy')
        }
        self.model_optimizations = {
            'pool_v_stride': 'stride',
            'dilated': False,
            'skip': None,  # None, dense, residual
            'multiscale': True,
            'initialize_trained': True
        }

        use_checkpoint = False
        if use_checkpoint:
            self.model_name = 'skip_res_small_conv_deconv_2017_08_30_12_06_56'
            self.ckpt_file = None
            self.resume_from_checkpoint = os.path.join(
                self.results_dir,
                self.model_name)
            if 'ckpt' in self.resume_from_checkpoint.split('/')[-1]:
                self.saved_config = '%s.npy' % os.path.sep.join(
                    self.resume_from_checkpoint.split('/')[:-1])
            else:
                self.saved_config = '%s.npy' % self.resume_from_checkpoint
        else:
            self.resume_from_checkpoint = None

        # Training settings
        self.epochs = 100
        self.debug = True  # Add gradient histograms
        self.feature_model_type = 'lakshmi_vgg_features'
        self.decision_model_type = 'lakshmi_vgg_decisions'
        self.fine_tune_layers = None
        self.batch_norm = [None]  # ['fc6', 'fc7', 'pre_fc8']
        self.train_batch = 32
        self.validation_batch = 32
        self.ratio = None  # [0.1, 0.9]
        self.lr = 3e-4  # Tune this -- also try SGD instead of ADAm
        self.hold_lr = 1e-8
        self.keep_checkpoints = 100
        self.optimizer = 'nadam'
        self.steps_before_validation = 1000
        self.loss_type = 'l2'
        self.grad_clip = False
        self.dim_weight = [1, 1, 0.5]   # If including xyz dim-specific weighting
        # MIGHT BE DEPRECIATED:
        self.use_training_loss = False  # early stopping based on loss
        self.early_stopping_rounds = 100
        self.test_proportion = 0.1  # TEST_RATIO
        self.mean_file = 'mean_file'  # Double check: used in training?
        self.selected_joints = None  # ['lThigh', 'lShin', 'lFoot', 'lToe', 'lToeMid3']  #  None  # ['lEye']  # Set to None to ignore
        self.num_dims = 3  # How many joint coordinate dimensions
        self.keep_dims = 3  # How many joint coordinate dimensions to optimize
        self.num_classes = 23 * self.num_dims
        self.mask_occluded_joints = False

        # Auxillary training settings
        self.normalize_labels = True
        self.aux_losses = []  # , 'deconv_label']  # ['z', 'size', 'occlusion', 'deconv_label']  # 'occlusion' 'pose' 'size' 'z' 'deconv_label' 'deconv'
        self.calculate_per_joint_loss = 'skeleton and joint pearson'  # 'skeleton and joint'  # TODO CLEAN UP THIS API
        self.wd_type = 'l2'
        self.wd_penalty = 5e-7
        self.wd_layers = [
            'high_feature_encoder_1x1_0',
            'high_feature_encoder_1x1_1',
            'high_feature_encoder_1x1_2'
        ]

        # Data augmentations
        self.data_augmentations = [
            'left_right',
            # 'up_down'
        ]
        self.background_multiplier = 1.  # Where to place the imaginary wall in the renders w.r.t. the max depth value
        self.randomize_background = None
        self.augment_background = 'constant'  #  'background'  # 'background_perlin'  # 'background_perlin'  # 'background_perlin'  # 'perlin'  # 'rescale' 'perlin' 'constant' 'rescale_and_perlin'
        self.background_folder = 'backgrounds'

        # Labels for the rendered images
        self.lakshmi_order = [
            100,
            97,
            57,
            60,
            79,
            61,
            80,
            62,
            81,
            69,
            91,
            71,
            93,
            38,
            19,
            39,
            20,
            40,
            21,
            41,
            22,
            50,
            31
        ]
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
