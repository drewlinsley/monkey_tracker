"""
Dictionary for BABAS-annotated frames.
"""

import os


def data():
        # {  # FOR DEBUGGING
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_monkey_tracking_data_for_babas_processed_videos_monkey_on_pole_2.npz'),
        #     'data': 'monkey_on_pole_3'
        # },
    return [
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_monkey_tracking_data_for_babas_processed_videos_monkey_on_pole_3_p1.npz'),
        #     'data': 'monkey_on_pole_3',
        #     'cv': 'validation'
        # },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_monkey_tracking_data_for_babas_processed_videos_monkey_in_cage_1_p1.npz'),
            'data': 'monkey_in_cage_1',
            'cv': 'train'
        },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_monkey_tracking_data_for_babas_processed_videos_monkey_in_cage_1_p2.npz'),
            'data': 'monkey_in_cage_1',
            'cv': 'train'
        },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_monkey_tracking_data_for_babas_processed_videos_monkey_in_cage_1_p3.npz'),
            'data': 'monkey_in_cage_1',
            'cv': 'train'
        },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_monkey_tracking_data_for_babas_processed_videos_monkey_in_cage_1_p4.npz'),
            'data': 'monkey_in_cage_1',
            'cv': 'train'
        },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_monkey_tracking_data_for_babas_processed_videos_monkey_in_cage_1_p5.npz'),
            'data': 'monkey_in_cage_1',
            'cv': 'train'
        },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_wisel_0.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_wisel_2.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_wisel_3.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_wisel_4.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_wisel_5.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_posner_0.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_giovanni_0_fix.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_x9_0.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # # #TODO: Get data from g18 and x7
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_x7_0.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0',
        #     'cv': 'train'
        # },
        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_Freely_Moving_Recording_depth_1_g3.npz'),
        #     'data': 'Freely_Moving_Recording_depth_1',
        #     'cv': 'train'
        # },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_Freely_Moving_Recording_depth_1_g3_test.npz'),
            'data': 'Freely_Moving_Recording_depth_1',
            'cv': 'validation'
        },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_Freely_Moving_Recording_depth_1_g3.npz'),
            'data': 'Freely_Moving_Recording_depth_1',
            'cv': 'train'
        },
        {
            'project': os.path.join(
                '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
                'babas_freely_moving_recording_0_posner.npz'),
            'data': 'Freely_Moving_Recording_depth_0',
            'cv': 'train'
        },


        # {
        #     'project': os.path.join(
        #         '/media/data_cifs/monkey_tracking/data_for_babas/babas_annotations',
        #         'babas_starbuck_pole_new_configuration_competition_depth_0_g18_0.npz'),
        #     'data': 'starbuck_pole_new_configuration_competition_depth_0'
        # },
    ]
