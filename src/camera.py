import cv2
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Camera(object):
    """ Camera """

    __CALIBRATED_CAMERA_MATRIX_PATH = "src/resources/new_dumps/camera_matrix.npy"
    __DISTORTION_COEF_PATH = "src/resources/new_dumps/distortion.npy"
    __DISTANCE = 60

    def __init__(self):
        self.t = [] # translation matrix
        self.R = [] # roatation matrix
        self.K = [] # calibration matrix

        self.distortion = [] # distortion coefficents

        self.F = [] # fundamential matrix
        self.E = [] # essential matrix
        
        self.__calibrate_camera()

    def project(self, obj):
        """ project 3d object into image coordinates """
        #R_t = np.hstack((self.R, self.t))
        #projection_matrix = np.dot(self.K, R_t) 
        #return np.dot(projection_matrix, obj)
        return cv2.projectPoints(
            obj,
            cv2.Rodrigues(self.R)[0],
            self.t,
            self.K,
            self.distortion
        )[0]

    def __calibrate_camera(self):
        """ Loads previously saved calibration matrix from file """
        self.distortion = np.load(self.__DISTORTION_COEF_PATH)
        self.K = np.load(self.__CALIBRATED_CAMERA_MATRIX_PATH)

    def compute_camera_extrinsic(self, image1, image2):
        orb = cv2.ORB_create()

        # get key points and descriptors
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x: x.distance)
        good = matches[:int(len(matches) * 0.1)]

        # select apropriate matches
        pts1 = []
        pts2 = []
        for m in matches[:int(len(matches) * 0.2)]:
            #if m.distance < self.__DISTANCE:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        # sift = cv2.xfeatures2d.SIFT_create()

        # kp1, des1 = sift.detectAndCompute(image1,None)
        # kp2, des2 = sift.detectAndCompute(image2,None)

        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks=50)

        # flann = cv2.FlannBasedMatcher(index_params,search_params)
        # matches = flann.knnMatch(des1,des2,k=2)

        # good = []
        # pts1 = []
        # pts2 = []

        # for i,(m,n) in enumerate(matches):
        #     if m.distance < 0.6*n.distance:
        #         good.append(m)
        #         pts2.append(kp2[m.trainIdx].pt)
        #         pts1.append(kp1[m.queryIdx].pt)

        # pts1 = np.int32(pts1)
        # pts2 = np.int32(pts2)

        # caluclate fundamential matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2, pp=(486., 265.))
        #points, R, t, mask = cv2.recoverPose(F, pts1, pts2, pp=(486., 265.), mask=mask);
        # E = np.dot(np.dot(self.K.T, F), self.K)
        # points, R, t, mask = cv2.recoverPose(E, pts1, pts2)

        # # SVD
        # U, s, V = np.linalg.svd(E, full_matrices=True)
        # W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # rotation = np.dot(np.dot(U, W.T), V.T)
        # translation = U[:, 2]

        self.E = E
        self.F = F
        self.R = R
        self.t = t

        
        # NOTE: for debug
        # img3 = cv2.drawMatches(image1,kp1,image2,kp2, good, None, flags=2)
        # cv2.imshow("asdasd", img3)

        #print(self.K)

        # homograpy
        M_r = np.hstack((self.R, self.t))
        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

        P_l = np.dot(self.K,  M_l)
        P_r = np.dot(self.K,  M_r)
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # for p in point_3d:
        #     ax.scatter(p[0], p[1], p[2], c='b', marker='^')

        # plt.show()


        print(point_3d)

        return E, F, R, t

