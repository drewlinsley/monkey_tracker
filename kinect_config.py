import os


class kinectConfig():

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self):
        # self.selected_video = 'starbuck_pole_new_configuration_competition_depth_0'
        self.selected_video = 'Freely_Moving_Recording_depth_0'
        self.defaults = {
            'rotate_frames': -1,
            'use_tfrecords': True,  # Package into tfrecords

            # Video frame and background subtraction params
            'start_frame': 100,
            'end_frame': 35,
            'low_threshold': None,
            'high_threshold': None,
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
            'crop': None,  # 'static',  # static or box
            'w': 175,
            'h': 150,
            '_x': 32,
            '_y': 40,
            'x_': 412,
            'y_': 300,
            'ignore_border_px': 10,

            # auto cnn bb
            'mask_with_model': False,
            'crop_and_pad': False,
            'small_object_size': 600,
            'output_joint_dict': False,

            # Normalization
            'max_adjust': 1,
            'min_adjust': 1,
            'kinect_max_adjust': 1,
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
        container['mask_with_model'] = False
        container['high_threshold'] = 3350  # Keep
        container['cnn_threshold'] = 50  # Keep
        container['crop_and_pad'] = False
        container['run_gmm'] = False
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
        container['background_mask'] = {
            'left': 140,
            'right': 0
        }        
        return container

    def monkey_in_cage_1(self):
        container = self.defaults
        container['output_joint_dict'] = False
        container['crop'] = 'static_and_crop'  # static or box
        container['start_frame'] = 100
        container['end_frame'] = 35
        container['h0'] = 180
        container['w0'] = 110
        container['h1'] = 325
        container['w1'] = 420
        container['cnn_threshold'] = 50  # Keep
        container['crop_and_pad'] = False
        container['mask_with_model'] = False
        container['run_gmm'] = False
        container['time_threshold'] = 95
        container['crop_and_pad'] = False
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
        container['background_mask'] = {
            'left': 140,
            'right': 0
        }        
        return container

    def monkey_in_cage_2(self):
        container = self.defaults
        container['output_joint_dict'] = False
        container['crop'] = 'static_and_crop'  # static or box
        container['start_frame'] = 100
        container['end_frame'] = 35
        container['h0'] = 180
        container['w0'] = 110
        container['h1'] = 325
        container['w1'] = 420
        container['cnn_threshold'] = 50  # Keep
        container['crop_and_pad'] = False
        container['find_bb'] = False
        container['mask_with_model'] = False
        container['run_gmm'] = False
        container['time_threshold'] = 95
        container['crop_and_pad'] = False
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            'predicted_monkey_in_cage_2')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'monkey_in_cage_2.mp4')
        container['output_json_path'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_jsons',
            'monkey_in_cage_2.json')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'predicted_monkey_in_cage_2.mp4')
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
        container['background_mask'] = {
            'left': 140,
            'right': 0
        }
        return container


    # New video extractions
    def xef2mat_output(self):
        container = self.defaults
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth',
            'Xef2Mat_Output')
        container['kinect_project'] = 'DepthFrame_npys'
        container['kinect_file_ext'] = '.npy'
        container['background_mask'] = {
            'left': 1300,
            'right': 1350
        }
        return container

    def xef2mat_output2(self):
        container = self.defaults
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth')
        container['kinect_project'] = 'Xef2Mat_Output_Trial02_np_conversion'
        container['kinect_file_ext'] = '.npy'
        container['background_mask'] = {
            'left': 500,
            'right': 10
        }
        return container

    def xef2mat_output4(self):
        container = self.defaults
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth',
            'Xef2Mat_Output_Trial04',
            'Xef2Mat_Output')
        container['kinect_project'] = 'DepthFrame_npys'
        container['kinect_file_ext'] = '.npy'
        container['background_mask'] = {
            'left': 1350,
            'right': 100
        }
        return container

    def starbuck_pole_new_configuration_competition_depth_0(self):
        """Depth frames. Pass through CNN."""
        container = self.defaults
        container['output_joint_dict'] = True
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth',
            'starbuck_pole_new_configuration_competition_depth_0')
        container['kinect_project'] = 'DepthFrame_npys'
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            container['kinect_project'])
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'starbuck_pole_new_configuration_competition_depth_0.mp4')
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'starbuck_pole_new_configuration_competition_depth_0.tfrecords')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'starbuck_pole_new_configuration_competition_depth_0.mp4')
        container['output_json_path'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_jsons',
            'starbuck_pole_new_configuration_competition_depth_0.json')
        container['output_npy_path'] = container['data_dir']
        container['kinect_file_ext'] = '.npy'
        return container

    def starbuck_pole_new_configuration_competition_IR_0(self):
        """IR frames. Don't pass through CNN."""
        container = self.defaults
        container['output_joint_dict'] = False
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth',
            'starbuck_pole_new_configuration_competition_IR_0')
        container['kinect_project'] = 'IRFrame_npys'
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            container['kinect_project'])
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'starbuck_pole_new_configuration_competition_IR_0.tfrecords')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'starbuck_pole_new_configuration_competition_IR_0.mp4')
        container['output_npy_path'] = container['data_dir']
        container['kinect_file_ext'] = '.npy'
        return container

    def trimmed_starbuck_pole_new_configuration_competition_depth_0(self):
        """Depth frames. Pass through CNN."""
        container = self.defaults
        container['output_joint_dict'] = False
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth')
        container['crop'] = 'static'  # static or box
        container['kinect_project'] = 'starbuck_pole_new_trim_depth_0'
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            container['kinect_project'])
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            'trimmed_starbuck_pole_new_configuration_competition_depth_0.mp4')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'starbuck_pole_new_trim_depth_0.tfrecords')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'starbuck_pole_new_trim_depth_0.mp4')
        container['gt_output_name'] = os.path.join(
            container['data_dir'],
            'gt_starbuck_pole_new_trim_depth_0.mp4')
        container['output_json_path'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_jsons',
            'starbuck_pole_new_trim_depth_0.json')
        container['output_npy_path'] = container['data_dir']
        container['kinect_file_ext'] = '.npy'
        return container

    def trimmed_starbuck_pole_new_configuration_competition_IR_0(self):
        """IR frames. Don't pass through CNN."""
        container = self.defaults
        container['output_joint_dict'] = False
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth')
        container['kinect_project'] = 'starbuck_pole_new_trim_IR_0'
        container['output_dir'] = '/home/drew/Desktop/'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            container['kinect_project'])
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            'starbuck_pole_new_trim_IR_0.tfrecords')
        container['kinect_output_name'] = os.path.join(
            '/media/data_cifs/monkey_tracking/data_for_babas/processed_videos',
            'starbuck_pole_new_trim_IR_0.mp4')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['output_npy_path'] = container['data_dir']
        container['kinect_file_ext'] = '.npy'
        return container

    def Freely_Moving_Recording_depth_0(self):
        """Depth frames. Pass through CNN."""
        container = self.defaults
        container['output_joint_dict'] = True
        container['kinect_directory'] = os.path.join(
            container['base_dir'],
            'extracted_kinect_depth',
            '201710091108-Freely_Moving_Recording_depth_1')
        container['kinect_project'] = 'DepthFrame_npys'
        container['output_dir'] = '/home/drew/Desktop/lakshmi_files'
        container['data_dir'] = os.path.join(
            container['output_dir'],
            container['kinect_project'])
        container['prediction_image_folder'] = os.path.join(
            container['data_dir'],
            'prediction_frames')
        container['gt_image_folder'] = os.path.join(
            container['data_dir'],
            'gt_frames')
        container['predicted_output_name'] = os.path.join(
            container['data_dir'],
            '201710091108-Freely_Moving_Recording_depth_1.mp4')
        container['tfrecord_name'] = os.path.join(
            container['data_dir'],
            '201710091108-Freely_Moving_Recording_depth_1.tfrecords')
        container['kinect_output_name'] = os.path.join(
            container['output_dir'],
            '201710091108-Freely_Moving_Recording_depth_1.mp4')
        container['output_json_path'] = os.path.join(
            container['output_dir'],
            '201710091108-Freely_Moving_Recording_depth_1.json')
        container['output_npy_path'] = container['data_dir']
        container['kinect_file_ext'] = '.npy'
        return container


    def Freely_Moving_Recording_depth_1(self):
            """Depth frames. Pass through CNN."""
            container = self.defaults
            container['output_joint_dict'] = True
            container['kinect_directory'] = os.path.join(
                container['base_dir'],
                'extracted_kinect_depth',
                '201710091108-Freely_Moving_Recording_depth_1')
            container['kinect_project'] = 'DepthFrame_npys'
            container['output_dir'] = '/home/lakshmi/'
            container['data_dir'] = os.path.join(
                container['output_dir'],
                container['kinect_project'])
            container['prediction_image_folder'] = os.path.join(
                container['data_dir'],
                'prediction_frames')
            container['gt_image_folder'] = os.path.join(
                container['data_dir'],
                'gt_frames')
            container['predicted_output_name'] = os.path.join(
                container['data_dir'],
                'Freely_Moving_Recording_depth_201710091108_1.mp4')
            container['tfrecord_name'] = os.path.join(
                container['data_dir'],
                '201710091108-Freely_Moving_Recording_depth_1.tfrecords')
            container['kinect_output_name'] = os.path.join(
                container['output_dir'],
                'kinect',
                '201710091108-Freely_Moving_Recording_depth_1.mp4')
            container['output_json_path'] = os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/processed_jsons',
                '201710091108-Freely_Moving_Recording_depth_1.json')
            container['output_npy_path'] = '/media/data_cifs/lakshmi/monkey_tracker/'
            container['kinect_file_ext'] = '.npy'
            return container
