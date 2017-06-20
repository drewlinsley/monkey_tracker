import os


class kinectConfig():

    def __init__(self):
        self.selected_video = 'monkey_on_pole_1'
        self.defaults = {
             # Video frame and background subtraction params
             'trim_start': 100,
             'trim_end': 35,
             'low_threshold': 1400,
             'high_threshold': 3350,
             'show_threshold_results': False,

             # Background GMM params
             'run_gmm': False,
             'bgsub_wraps': 1,  # Set to None if you don't want this
             'bgsub_quorum': 1,
             'bgsub_mog_bg_theshold': 10,
             'show_mog_result': False,
             'left_frame': 100,
             'right_frame': 40,

             # Crop box params
             'crop': False,
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
        monkey_on_pole_1['kinect_output_name'] = None
        monkey_on_pole_1['predicted_output_name'] = os.path.join(
            monkey_on_pole_1['output_dir'],
            'predicted_monkey_on_pole_1.mp4')
        return monkey_on_pole_1

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)
