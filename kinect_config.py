import os


class kinectConfig():

    def __init__(self):
        self.selected_video = 'monkey_in_cage_1'  # 'monkey_on_pole_3'  # 
        self.defaults = {
            'rotate_frames': -1,
            'use_tfrecords': True,  # Package into tfrecords

            # Video frame and background subtraction params
            'start_frame': 100,
            'end_frame': 35,
            'low_threshold': 1400,
            'high_threshold': 3350,
            'show_threshold_results': False,

            # Bounding box crop the monkey
            'find_bb': False,
            'time_threshold': 95,  # time-masking

            # Background GMM params
            'run_gmm': False,
            'bgsub_wraps': 1,  # Set to None if you don't want this
            'bgsub_quorum': 1,
            'bgsub_mog_bg_theshold': 10,
            'show_mog_result': False,
            'left_frame': 100,
            'right_frame': 40,

            # Crop box params
            'crop': 'static',  # static or box
            'w': 175,
            'h': 150,
            '_x': 32,
            '_y': 40,
            'x_': 412,
            'y_': 300,
            'ignore_border_px': 10,

            # auto cnn bb
            'mask_with_model': True,
            'crop_and_pad': True,
            'small_object_size': 600,

            # Normalization
            'max_adjust': 0.5,
            'min_adjust': 2,
            'kinect_max_adjust': 0.75,
            'kinect_min_adjust': 1,

            # Kinect file settings
            'base_dir': '/media/data_cifs/monkey_tracking/',
        }

    def monkey_on_pole_1(self):
        container = self.defaults
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            'predicted_monkey_on_pole_1')
        container['kinect_output_name'] = os.path.join(
            container['data_dir'],
            'vid_int_movie.mp4')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'predicted_monkey_on_pole_1.mp4')
        container['gt_output_name'] = os.path.join(
            container['data_dir'],
            'gt_monkey_on_pole_1.mp4')
        container['output_npy_path'] = container['data_dir']
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'monkey_on_pole.tfrecords')
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')

        container['kinect_directory'] = os.path.join(
                container['base_dir'],
                'extracted_kinect_depth')
        container['kinect_project'] = 'Xef2Mat_Output_Trial02_np_conversion'
        container['kinect_file_ext'] = '.npy'
        container['kinect_video'] = 'video.mp4'
        return container

    def monkey_on_pole_2(self):
        container = self.defaults
        container['time_threshold'] = 95
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            'predicted_monkey_on_pole_2')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'vid_int_movie.mp4')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'predicted_monkey_on_pole_2.mp4')
        container['gt_output_name'] = None  # os.path.join(
        container['output_npy_path'] = container['data_dir']
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'monkey_on_pole.tfrecords')
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['kinect_directory'] = os.path.join(
                container['base_dir'],
                'extracted_kinect_depth')
        container['kinect_project'] = 'Xef2Mat_Output_Trial02_np_conversion'
        container['kinect_file_ext'] = '.npy'
        container['kinect_video'] = 'video.mp4'
        container['max_adjust'] = 0.5
        container['min_adjust'] = 2
        return container

    def monkey_on_pole_3(self):
        container = self.defaults
        container['start_frame'] = 150  # Keep
        container['end_frame'] = 35  # Keep
        container['time_threshold'] = 80
        container['find_bb'] = False  # Do not use when outputting moies
        container['low_threshold'] = 1400  # Keep
        container['high_threshold'] = 3350  # Keep
        container['cnn_threshold'] = 60  # Keep
        container['crop_and_pad'] = False
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            'predicted_monkey_on_pole_3')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'monkey_on_pole_3.mp4')
        container['output_json_path'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_jsons',
            'monkey_on_pole_3.json')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'predicted_monkey_on_pole_3.mp4')
        container['gt_output_name'] = None  # os.path.join(
        container['output_npy_path'] = container['data_dir']
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'monkey_on_pole.tfrecords')
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['kinect_directory'] = os.path.join(
                container['base_dir'],
                'extracted_kinect_depth')
        container['kinect_project'] = 'Xef2Mat_Output_Trial02_np_conversion'
        container['kinect_file_ext'] = '.npy'
        container['kinect_video'] = 'video.mp4'
        container['max_adjust'] = 1
        container['min_adjust'] = 1
        container['kinect_max_adjust'] = 1
        container['kinect_min_adjust'] = 1
        return container

    def monkey_in_cage_1(self):
        container = self.defaults
        container['crop'] = 'static_and_crop'  # static or box
        container['start_frame'] = 100
        container['end_frame'] = 35
        container['h0'] = 180
        container['w0'] = 105
        container['h1'] = 395
        container['w1'] = 420
        container['time_threshold'] = 95
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            'predicted_monkey_in_cage_1')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'monkey_in_cage_1.mp4')
        container['output_json_path'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_jsons',
            'monkey_in_cage_1.json')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'predicted_monkey_in_cage_1.mp4')
        container['gt_output_name'] = None
        container['output_npy_path'] = container['data_dir']
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'monkey_on_pole.tfrecords')
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['kinect_directory'] = os.path.join(
                container['base_dir'],
                'extracted_kinect_depth',
                'Xef2Mat_Output_Trial04',
                'Xef2Mat_Output')
        container['kinect_project'] = 'DepthFrame_npys'
        container['kinect_file_ext'] = '.npy'
        container['kinect_video'] = 'video.mp4'
        return container

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)
