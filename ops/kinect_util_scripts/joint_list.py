"""
Utility class for mapping joint names and their connectivity graph.
"""


class joint_data(object):
    """
    Class for mapping joint names and their connectivity graph.
    """
    def __init__(self):
        """Store information in the class init."""
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
        self.joint_connections = [
            'head',
            'neck',
            'left shoulder',
            'left elbow',
            'left hand',
            'left knuckles',
            'left fingertips',
            'left knuckles',
            'left hand',
            'left elbow',
            'left shoulder',
            'neck',
            'right shoulder',
            'right elbow',
            'right hand',
            'right knuckles',
            'right fingertips',
            'right knuckles',
            'right hand',
            'right elbow',
            'right shoulder',
            'neck',
            'abdomen',
            'right hip',
            'right knee',
            'right ankle',
            'right bart',
            'right toetips',
            'right bart',
            'right ankle',
            'right knee',
            'right hip',
            'abdomen',
            'left hip',
            'left knee',
            'left ankle',
            'left bart',
            'left toetips',
            'left bart',
            'left ankle',
            'left knee',
            'left hip',
            'abdomen',
            'neck',
            'head'
        ]
