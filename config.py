from os.path import join as pjoin


#  Main configuration file for monkey tracking
class monkeyConfig(object):
    def __init__(self): 

        # Directory settings
        self.base_dir = '/media/data_cifs/monkey_tracking/' #'/media/data_cifs/monkey_tracking/batches/MovieRender'
        self.results_dir = pjoin(self.base_dir, 'results', 'TrueDepth100kStore') #'/media/data_cifs/monkey_tracking/batches/MovieRender'
        self.image_dir = pjoin(self.base_dir, 'batches', 'TrueDepth100kStore') #pjoin(self.base_dir, 'walk-all-png') 
        self.depth_dir = pjoin(self.image_dir, 'depth', 'true_depth')
        self.label_dir = pjoin(self.image_dir, 'labels', 'joint_coords')
        self.npy_dir = pjoin(self.base_dir, 'batches', 'test')  # Output for numpys
        self.pixel_label_dir = pjoin(self.image_dir, 'labels', 'pixel_joint_coords')  # 'joint_coords')
        self.occlusion_dir = pjoin(self.image_dir, 'labels', 'occlusions')  # Set to None if there's no occlusion data
        self.im_label_dir = pjoin(self.image_dir, 'labels', 'npyLabels')
        self.depth_regex = '*[0-9].npy'
        self.image_extension = '.npy'
        self.label_extension = '.npy' 
        self.occlusion_extension = '.npy'
        self.model_output = pjoin(self.results_dir, 'model_output') 
        self.tfrecord_dir = pjoin(self.image_dir, 'tfrecords_drew')
        self.train_summaries = pjoin(self.results_dir, 'summaries')
        self.train_checkpoint = pjoin(self.results_dir, 'checkpoints')
        self.vgg16_weight_path = pjoin(
            '/media/data_cifs/clicktionary/',
            'pretrained_weights',
            'vgg16.npy')
        self.resume_from_checkpoint = None  # '/media/data_cifs/monkey_tracking/batches/CnnMultiLowHigh2/walk-all-png/model_output/cnn_multiscale_low_high_res_2017_05_22_14_59_44/model_31600.ckpt-31600'

        # Tfrecords
        self.new_tf_names = {'train': 'train.tfrecords' , 'val': 'val.tfrecords'}  # {'train': 'train_2mill.tfrecords', 'val': 'val_2mill.tfrecords'}
        self.train_tfrecords = 'train.tfrecords'  # Decouple the these vars so you can create new records while training #'train_2mill.tfrecords' 
        self.val_tfrecords = 'val.tfrecords'  # 'val_2mill.tfrecords'        
        self.max_train = None  # Limit the number of files we're going to store in a tfrecords. Set to None if there's no limit.
        self.max_depth = 1300.  # Divide each image by this value to normalize it to [0, 1]. This is the only normalization we will do. Must be a float!
        self.background_constant = self.max_depth * 2  # HIGH_NUMBER
        self.resize = [240, 320, 3]  # CNN input (don't change) -- make sure this the same dimensions as the input
        self.image_input_size = [480, 640]  # Maya render output
        self.image_target_size = [240, 320, 3]  # Resize before tfrecords
        self.maya_conversion = 640.0 / 500.0  # pixels / maya units
        self.sample = {'train': True, 'val': False}  # random sample of feats
        self.use_image_labels = False  # if true, extract  color-labeled images
        self.use_pixel_xy = True

        # Model settings
        self.epochs = 50
        self.model_type = 'cnn_multiscale_high_res_low_res_skinny_pose_occlusion'
        self.initialize_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        self.fine_tune_layers = ['fc6', 'fc7', 'pre_fc8', 'fc8']
        self.batch_norm = ['fc6', 'fc7', 'pre_fc8']
        self.data_augmentations = [
            'convert_labels_to_pixel_space',  # commented out bc we want to train the model on the 3D coordinates, not pixel positions
        ]

        # Key training settings
        self.train_batch = 64
        self.validation_batch = 1
        self.ratio = None  # [0.1, 0.9]
        self.lr = 1e-4  # Tune this -- also try SGD instead of ADAm
        self.hold_lr = 1e-4
        self.keep_checkpoints = 100
        self.optimizer = 'adam'
        self.steps_before_validation = 1000
        self.loss_type = 'l1'

        # Potentially outdated training settings
        self.use_training_loss = False  # early stopping based on loss
        self.early_stopping_rounds = 100
        self.test_proprtion = 0.1  # TEST_RATIO
        self.mean_file = 'mean_file'  # Double check: used in training?

        # Auxillary training settings
        self.normalize_labels = True  # True
        self.aux_losses = ['occlusion']  # ['fc', 'occlusion']  # , 'pose']  #  (i.e. angle between neck and abdomen)
        self.calculate_per_joint_loss = True
        self.include_validation = True
        self.wd_type = 'l1'
        self.wd_penalty = None  # 5e-4
        self.wd_layers = ['occlusion', 'output']  # ['fc6', 'fc7', 'pre_fc8']
        self.fc_lambda = 0.01

        # Kinect file settings
        self.kinect_directory = pjoin(self.base_dir, 'extracted_kinect_depth')
        self.kinect_project = 'Xef2Mat_Output_Trial02_np_conversion'
        self.kinect_file_ext = '.npy'
        self.kinect_video = 'video.mp4'

        # Labels for the rendered images
        self.labels = {
            'back_torso':      (99,  130,   0, 254),
            'background':      (0,     0,   0,   0),
            'front_torso':     (250, 200, 189, 254),
            'head':            (199,   0,   0, 254),
            'hip':             (130,  50, 120, 254),
            'left_finger':     (200,   0, 200, 254),
            'left_foot':       (150, 130, 139, 254),
            'left_hand':       (250,   0, 250, 254),
            'left_lower_arm':  (0,   250,   0, 254),
            'left_lower_leg':  (0,     0, 250, 254),
            'left_shoulder':   (50,   90, 139, 254),
            'left_toe':        (170, 190,  39, 254),
            'left_upper_arm':  (249, 180,   0, 254),
            'left_upper_leg':  (240, 240, 239, 254),
            'neck':            (70,   70,  69, 254),
            'right_finger':    (249,  50,   0, 254),
            'right_foot':      (0,   150,   0, 254),
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
            'neck'
            'abdomen',
            'abdomen2',
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
            'left foot',
            'right foot',
            'left toetips',
            'right toetips'
        ]
        self.selected_joints = None  # ['lEye']  # Set to None to ignore
        self.num_dims = 3
        self.keep_dims = 2
        self.num_classes = len(self.joint_order) * self.num_dims
        self.mask_occluded_joints = False

        # Feature extraction settings for classic kinect alg
        self.offset_nn = 30  # random +/- x,y pixel offset range # Tune this
        self.n_features = 400  # Tune this 
        self.max_pixels_per_image = 800  # Tune this
        self.cte_depth = 2  # ?? 

