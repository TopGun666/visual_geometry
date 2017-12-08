import cv2
import numpy as np

class Camera(object):
    """ Camera """

    __CALIBRATED_CAMERA_MATRIX_PATH = "src/resources/camera_matrix.npy"
    __DISTANCE = 60

    def __init__(self):
        self.t = [] # translation matrix
        self.R = [] # roatation matrix
        self.K = [] # calibration matrix

        self.F = [] # fundamential matrix
        self.E = [] # essential matrix

        self.__calibrate_camera()

    def project(self, obj):
        """ project 3d object into image coordinates """
        R_t = np.hstack((self.R, self.t))
        projection_matrix = np.dot(self.K, R_t) 

        return np.dot(projection_matrix, obj)

    def __calibrate_camera(self):
        """ Loads previously saved calibration matrix from file """
        self.K = np.load(self.__CALIBRATED_CAMERA_MATRIX_PATH)

    def compute_camera_extrinsic(self, image1, image2):
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
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2)

        self.E = E
        self.F = F
        self.R = R
        self.t = t

        return E, F, R, t