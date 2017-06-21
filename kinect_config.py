import os
from visualization.preprocess_kinect import \
    get_and_trim_frames, static_background, threshold, \
    bgsub_frames, box_tracking, multianimate


class KinectConfig():

    defaults = {
        # Video trimming
        'trim_start': 100,
        'trim_end': 35,

        # Thresholding
        'do_thresholding': True,
        'low_threshold': 1400,
        'high_threshold': 3350,
        'denoise_threshold_mask': True,
        # Display after threshold
        'show_threshold_results': False,

        # Background GMM params
        'do_bgsub': True,
        'bgsub_wraps': 1,
        'bgsub_quorum': 1,
        'bgsub_mog_bg_theshold': 10,
        'num_openings_and_closings': 3,
        'left_frame': 100,
        'right_frame': 40,
        # Display after GMM
        'show_mog_result': False,

        # Crop box params
        'do_box_tracking': True,
        'w': 175,
        'h': 150,
        '_x': 32,
        '_y': 40,
        'x_': 412,
        'y_': 300,
        'ignore_border_px': 10,

        # General process parameters
        # where is the data?
        'input_dir': '/media/data_cifs/monkey_tracking/somesubfolder/',
        # output processed frames as .tfrecords to `output_dir`/`name`/
        'save_tfrecords_to_disk': False,
        'output_dir': '~/monkey_data/',
        'name': 'monkey_video_name',
        'display_final_results': True,
    }


    @classmethod
    def monkey_on_pole_1(cls):
        config = cls.defaults
        config['name'] = 'monkey_on_pole_1'
        config['input_dir'] = '/media/data_cifs/monkey_tracking/extracted_kinect_depth/Xef2Mat_Output_Trial02_np_conversion/'
        return config


    @classmethod
    def monkey_on_pole_2(cls):
        config = cls.defaults
        config['name'] = 'monkey_on_pole_2'
        config['input_dir'] = '/media/data_cifs/monkey_tracking/extracted_kinect_depth/Xef2Mat_Output_Trial03_np_conversion/'
        return config


    @staticmethod
    def process(configuration):
        '''
        process frames according to `configuration`,
        a config dict like defaults.
        it gets a little ugly since there are a lot of combinations
        of actions that the config might specify.
        '''
        c = configuration
        print('Processing %s...' % c['name'])
        # load up frames
        final = frames = get_and_trim_frames(c['input_dir'], c['trim_start'], c['trim_end'])
        d = c['display_final_results']
        if d: alolom, titles = [frames], ['Original']

        # if we're subtracting the background, make a synthetic static
        # background to prime the gmm
        if c['do_bgsub']:
            bg = static_background(frames)
            frames = [bg] + frames

        # do thresholding and background subtraction
        if c['do_thresholding']:
            threshed = threshold(frames, c['low_threshold'], c['high_threshold'],
                                 show_result=c['show_threshold_results'],
                                 denoise=c['denoise_threshold_mask'])
            final = threshed
            if c['do_bgsub']:
                res = bgsub_frames(threshed, c['bgsub_wraps'], c['bgsub_quorum'],
                                   c['show_mog_result'], c['bgsub_mog_bg_theshold'],
                                   c['num_openings_and_closings'])
                if d:
                    alolom.append(threshed[1:]); titles.append('Thresholded')
                    alolom.append(res[1:]); titles.append('GMM')
            elif d:
                alolom.append(threshed); titles.append('Thresholded')
        elif c['do_bgsub']:
            res = bgsub_frames(frames, c['bgsub_wraps'], c['bgsub_quorum'],
                               c['show_mog_result'], c['bgsub_mog_bg_theshold'],
                               c['num_openings_and_closings'])
            if d: alolom.append(res[1:]); titles.append('GMM')

        # handle all of the cases for box tracking
        final = res
        if c['do_box_tracking']:
            btargs = (c['w'], c['h'], c['_x'], c['_y'], c['x_'], c['y_'])
            if c['do_bgsub'] and c['do_thresholding']:
                final = box_tracking(threshed[1:], *btargs, binaries=res[1:], 
                                     ignore_border_px=c['ignore_border_px'])
            elif c['do_bgsub']:
                final = box_tracking(frames[1:], *btargs, binaries=res[1:], 
                                     ignore_border_px=c['ignore_border_px'])
            elif c['do_thresholding']:
                final = box_tracking(threshed, *btargs, 
                                     ignore_border_px=c['ignore_border_px'])
            else:
                final = box_tracking(frames, *btargs, 
                                     ignore_border_px=c['ignore_border_px'])

        if d:
            alolom.append(final); titles.append('Result')
            multianimate(alolom, titles, c['name'])

        return final



    @classmethod
    def process_all_videos(cls):
        configs = [cls.monkey_on_pole_1, cls.monkey_on_pole_2]
        return [cls.process(cfg()) for cfg in configs]

if __name__ == '__main__':
    # KinectConfig.process_all_videos()
    KinectConfig.process(KinectConfig.monkey_on_pole_2())
