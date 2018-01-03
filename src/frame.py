import cv2

import numpy as np 
import pandas as pd

from mystic.penalty import quadratic_inequality
from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor

from src.camera import Camera
from src.utils.reprojection import reprojection_distance

class Frame(object):
    """ Frame object that stores all possible information needed """
    MATCHER_THREHSOLD = 0.7
    
    def __init__(self, image):
        self.image = np.copy(image)
        self.R = np.array([])
        self.t = np.array([])
        self.points_3d = np.array([])
        self.points_2d = np.array([])
        self.kp = np.array([])
        self.des = np.array([])
        
    def __str__(self):
        return "Frame:\n>> R:{}\n>> t:{}\n>> 3D:{}\n>> 2D:{}\n>> kp:{}\n>> des:{}\n".format(
            self.R.shape,
            self.t.shape,
            self.points_3d.shape,
            self.points_2d.shape,
            len(self.kp),
            len(self.des)
        )

    def create_camera_and_project(self, obj):
        camera = Camera.create(self.R, self.t)
        return camera.project(obj)
    
    def bundle_adjustment(self):
        """ Bundle adjust frame """

        R = cv2.Rodrigues(self.R)[0]
        t = self.t
        initial = [R[0], R[1], R[2], t[0], t[1], t[2]]
        initial = [x[0] for x in initial]
        
        mon = VerboseMonitor(50)
        result = diffev2(
            reprojection_distance, 
            x0=initial, 
            args=(self.points_3d, self.points_2d),
            npop=10, 
            gtol=200,
            disp=False, 
            full_output=True, 
            itermon=mon, 
            maxiter=500
        )
        
        result = result[0] #optimized.x
        r1, r2, r3, t1, t2, t3 = result[0], result[1], result[2], result[3], result[4], result[5]

        R_optimized = cv2.Rodrigues(np.array([r1, r2, r3]))[0]
        t_optimized = np.array([[t1], [t2], [t3]])
        
        optimized_frame = Frame(self.image)
        optimized_frame.R = R_optimized
        optimized_frame.t = t_optimized
        optimized_frame.points_3d = self.points_3d
        optimized_frame.points_2d = self.points_2d
        optimized_frame.kp = self.kp
        optimized_frame.des = self.des
        
        return optimized_frame