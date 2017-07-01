import os
from visualization.preprocess_kinect import \
    get_and_trim_frames, static_background, threshold, \
    bgsub_frames, box_tracking, multianimate


# <<<<<<< HEAD
class kinectConfig():

    def __init__(self):
        self.selected_video = 'monkey_on_pole_2'
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
             'find_bb': True,
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
             'ignore_border_px': 10
        }

    def monkey_on_pole_1(self):
        monkey_on_pole_1 = self.defaults
        monkey_on_pole_1['output_dir'] = '/home/drew/Desktop/'
        monkey_on_pole_1['data_dir'] = os.path.join(
            monkey_on_pole_1['output_dir'],
            'predicted_monkey_on_pole_1')
        monkey_on_pole_1['kinect_output_name'] = os.path.join(
            monkey_on_pole_1['data_dir'],
            'vid_int_movie.mp4')
        monkey_on_pole_1['predicted_output_name'] = os.path.join(
            monkey_on_pole_1['data_dir'],
            'predicted_monkey_on_pole_1.mp4')
        monkey_on_pole_1['gt_output_name'] = os.path.join(
            monkey_on_pole_1['data_dir'],
            'gt_monkey_on_pole_1.mp4')
        monkey_on_pole_1['output_npy_path'] = monkey_on_pole_1['data_dir']
        monkey_on_pole_1['tfrecord_name'] = os.path.join(
            monkey_on_pole_1['data_dir'],
            'monkey_on_pole.tfrecords')
        monkey_on_pole_1['prediction_image_folder'] = os.path.join(
            monkey_on_pole_1['data_dir'],
            'prediction_frames')
        monkey_on_pole_1['gt_image_folder'] = os.path.join(
            monkey_on_pole_1['data_dir'],
            'gt_frames')
        return monkey_on_pole_1

    def monkey_on_pole_2(self):
        monkey_on_pole_2 = self.defaults
        monkey_on_pole_2['time_threshold'] = 95
        monkey_on_pole_2['output_dir'] = '/home/drew/Desktop/'
        monkey_on_pole_2['data_dir'] = os.path.join(
            monkey_on_pole_2['output_dir'],
            'predicted_monkey_on_pole_2')
        monkey_on_pole_2['kinect_output_name'] = os.path.join(
            monkey_on_pole_2['data_dir'],
            'vid_int_movie.mp4')
        monkey_on_pole_2['predicted_output_name'] = os.path.join(
            monkey_on_pole_2['data_dir'],
            'predicted_monkey_on_pole_2.mp4')
        monkey_on_pole_2['gt_output_name'] = os.path.join(
            monkey_on_pole_2['data_dir'],
            'gt_monkey_on_pole_2.mp4')
        monkey_on_pole_2['output_npy_path'] = monkey_on_pole_2['data_dir']
        monkey_on_pole_2['tfrecord_name'] = os.path.join(
            monkey_on_pole_2['data_dir'],
            'monkey_on_pole.tfrecords')
        monkey_on_pole_2['prediction_image_folder'] = os.path.join(
            monkey_on_pole_2['data_dir'],
            'prediction_frames')
        monkey_on_pole_2['gt_image_folder'] = os.path.join(
            monkey_on_pole_2['data_dir'],
            'gt_frames')
        return monkey_on_pole_2

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)
# =======

# class KinectConfig():

#     defaults = {
#         # Video trimming
#         'trim_start': 100,
#         'trim_end': 35,

#         # Thresholding
#         'do_thresholding': True,
#         'low_threshold': 1400,
#         'high_threshold': 3350,
#         'denoise_threshold_mask': True,
#         # Display after threshold
#         'show_threshold_results': False,

#         # Background GMM params
#         'do_bgsub': True,
#         'bgsub_wraps': 1,
#         'bgsub_quorum': 1,
#         'bgsub_mog_bg_theshold': 10,
#         'num_openings_and_closings': 3,
#         'left_frame': 100,
#         'right_frame': 40,
#         # Display after GMM
#         'show_mog_result': False,

#         # Crop box params
#         'do_box_tracking': True,
#         'w': 175,
#         'h': 150,
#         '_x': 32,
#         '_y': 40,
#         'x_': 412,
#         'y_': 300,
#         'ignore_border_px': 10,

#         # General process parameters
#         # where is the data?
#         'input_dir': '/media/data_cifs/monkey_tracking/somesubfolder/',
#         # output processed frames as .tfrecords to `output_dir`/`name`/
#         'save_tfrecords_to_disk': False,
#         'output_dir': '~/monkey_data/',
#         'name': 'monkey_video_name',
#         'display_final_results': True,
#     }


#     @classmethod
#     def monkey_on_pole_1(cls):
#         config = cls.defaults
#         config['name'] = 'monkey_on_pole_1'
#         config['input_dir'] = '/media/data_cifs/monkey_tracking/extracted_kinect_depth/Xef2Mat_Output_Trial02_np_conversion/'
#         return config


#     @staticmethod
#     def process(configuration):
#         '''
#         process frames according to `configuration`,
#         a config dict like defaults
#         '''
#         c = configuration
#         print('Processing %s...' % c['name'])
#         # load up frames
#         final = frames = get_and_trim_frames(c['input_dir'], c['trim_start'], c['trim_end'])
#         d = c['display_final_results']
#         if d: alolom, titles = [frames], ['Original']

#         if c['do_bgsub']:
#             bg = static_background(frames)
#             frames = [bg] + frames


#         if c['do_thresholding']:
#             threshed = threshold(frames, c['low_threshold'], c['high_threshold'],
#                                  show_result=c['show_threshold_results'],
#                                  denoise=c['denoise_threshold_mask'])
#             final = threshed
#             if c['do_bgsub']:
#                 res = bgsub_frames(threshed, c['bgsub_wraps'], c['bgsub_quorum'],
#                                    c['show_mog_result'], c['bgsub_mog_bg_theshold'],
#                                    c['num_openings_and_closings'])
#                 if d:
#                     alolom.append(threshed[1:]); titles.append('Thresholded')
#                     alolom.append(res[1:]); titles.append('GMM')
#             elif d:
#                 alolom.append(threshed); titles.append('Thresholded')
#         elif c['do_bgsub']:
#             res = bgsub_frames(frames, c['bgsub_wraps'], c['bgsub_quorum'],
#                                c['show_mog_result'], c['bgsub_mog_bg_theshold'],
#                                c['num_openings_and_closings'])
#             if d: alolom.append(res[1:]); titles.append('GMM')

#         # handle all of the cases for box tracking
#         final = res
#         if c['do_box_tracking']:
#             btargs = (c['w'], c['h'], c['_x'], c['_y'], c['x_'], c['y_'])
#             if c['do_bgsub'] and c['do_thresholding']:
#                 final = box_tracking(threshed[1:], *btargs, binaries=res[1:], 
#                                      ignore_border_px=c['ignore_border_px'])
#             elif c['do_bgsub']:
#                 final = box_tracking(frames[1:], *btargs, binaries=res[1:], 
#                                      ignore_border_px=c['ignore_border_px'])
#             elif c['do_thresholding']:
#                 final = box_tracking(threshed, *btargs, 
#                                      ignore_border_px=c['ignore_border_px'])
#             else:
#                 final = box_tracking(frames, *btargs, 
#                                      ignore_border_px=c['ignore_border_px'])

#         if d:
#             alolom.append(final); titles.append('Result')
#             multianimate(alolom, titles, c['name'])

#         return final



#     @classmethod
#     def process_all_videos(cls):
#         configs = [cls.monkey_on_pole_1]
#         return [cls.process(cfg()) for cfg in configs]

# if __name__ == '__main__':
#     KinectConfig.process_all_videos()
