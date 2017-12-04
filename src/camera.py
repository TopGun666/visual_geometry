import cv2
import numpy as np

class Camera(object):
    __CALIBRATED_CAMERA_MATRIX_PATH = "src/resources/dumps/camera_matrix.npy"
    __DISTANCE = 80

    """ Camera """
    def __init__(self):
        self.t = [] # translation matrix
        self.R = [] # roatation matrix
        self.K = [] # calibration matrix

        self.F = [] # fundamential matrix
        self.E = [] # essential matrix

        self.__calibrate_camera()

    def project(self, obj):
        """ project 3d object into image coordinates """
        raise NotImplementedError

    def __calibrate_camera(self):
        """ Loads previously saved calibration matrix from file """
        self.K = np.load(self.__CALIBRATED_CAMERA_MATRIX_PATH)

    def compute_fundamential_matrix(self, image1, image2):
        orb = cv2.ORB_create()

        # get key points and descriptors
        kp1, des1 = orb.detectAndCompute(image1,None)
        kp2, des2 = orb.detectAndCompute(image2,None)

        # run matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x: x.distance)
        
        # select apropriate matches
        pts1 = []
        pts2 = []
        for m in matches:
            if m.distance < self.__DISTANCE:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        # caluclate fundamential matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        self.F = F
        return F