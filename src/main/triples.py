import cv2
import numpy as np


class KeyFrame(object):
    """ Object for keyframe parameters """

    def __init__(self, frame, index):
        """ Constructor
        
        Args:
            frame: 2d numpy matrix with image
        """
        self.frame = frame
        self.frame_index = index
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
        self.matches = matching_points

    def points_match(self, im1, im2):
        """  """
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # get key points and descriptors
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)

        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x: x.distance)
        good = matches[:int(len(matches) * 0.1)]

        pts1 = []
        pts2 = []
        for m in self.matches[:10]:
            #if m.distance < self.__DISTANCE:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        return pts2

    def initial_stereo_reconstruction(self, K):
        """  """
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # get key points and descriptors
        kp1, des1 = orb.detectAndCompute(self.f1.frame, None)
        kp2, des2 = orb.detectAndCompute(self.f2.frame, None)

        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x: x.distance)
        good = matches[:int(len(matches) * 0.1)]

        pts1 = []
        pts2 = []
        for m in self.matches[:10]:
            #if m.distance < self.__DISTANCE:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2, pp=(486., 265.))

        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        M_r = np.hstack((R, t))

        P_l = np.dot(K,  M_l)
        P_r = np.dot(K,  M_r)
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T


        self.f1.R = np.eye(3, 3)
        self.f1.t = np.zeros((3, 1))

        self.f2.R = R
        self.f2.t = t

        self.triangulation = point_3d

        return point_3d

    def recover_third(self, K, dist):
        f2_image, f3_image = self.f2.frame, self.f3.frame
        image_points = self.points_match(f2_image, f3_image)

        print(image_points, self.triangulation)
        
        ret, rvecs, tvecs, inliers = cv2.solvePnP(self.triangulation, image_points, K, dist)

        self.f3.R = rvecs
        self.f3.t = tvecs






    