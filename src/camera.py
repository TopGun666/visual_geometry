import cv2
import numpy as np

class Camera(object):
    """ Camera """
    def __init__(self):
        self.P = [] # projection matrix
        self.t = [] # translation matrix
        self.R = [] # roatation matrix
        self.K = [] # calibration matrix

    def project(self, obj):
        """ project 3d object into image coordinates """
        raise NotImplementedError

    def calibrate_camera(self):
        """ Loads previously saved calibration matrix from file """
        raise NotImplementedError
