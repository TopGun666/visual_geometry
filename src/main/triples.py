import numpy


class KeyFrame(object):
    """ Object for keyframe parameters """

    def __init__(self, frame):
        """ Constructor
        
        Args:
            frame: 2d numpy matrix with image
        """
        self.frame = frame
        self.R = []
        self.t = []

class KeyFrameTriple(object):
    """ Object for keyframe triples logic """
    
    def __init__(
        self, 
        f1, 
        f2, 
        f3,
        matching_points
    ):
        """ Constructor

        Args:
            f1 (KeyFrame): key frame #1
            f2 (KeyFrame): key frame #2
            f3 (KeyFrame): key frame #3
            matching_points: points that are matches across three images
        """
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.matching_points = matching_points

    