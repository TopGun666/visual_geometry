import cv2
import numpy as np

class Camera(object):
    __CALIBRATED_CAMERA_MATRIX_PATH = "src/resources/dumps/camera_matrix.npy"

    """ Camera """
    def __init__(self):
        self.t = [] # translation matrix
        self.R = [] # roatation matrix
        self.K = [] # calibration matrix

        self.__calibrate_camera()

    def project(self, obj):
        """ project 3d object into image coordinates """
        raise NotImplementedError

    def __calibrate_camera(self):
        """ Loads previously saved calibration matrix from file """
        self.K = np.load(self.__CALIBRATED_CAMERA_MATRIX_PATH)

    