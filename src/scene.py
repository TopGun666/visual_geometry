import cv2

import numpy as np 
import pandas as pd

from src.camera import Camera
from src.utils.cube import generate_cube
from src.frame import Frame

class Scene(object):
    """ Scene utils """
    @classmethod
    def get_rt_from_essential(cls, pts1, pts2):
        E, mask = cv2.findEssentialMat(
            pts1, 
            pts2, 
            focal=1.0, 
            pp=(486.2, 265.59), 
            method=cv2.RANSAC, 
            prob=0.999, 
            threshold=1.0
        )

        points, R, t, mask = cv2.recoverPose(
            E, 
            pts1, 
            pts2, 
            focal=1.0,
            pp=(486.2, 265.59),
            mask=mask
        )
        
        return R, t
    
    
    @classmethod
    def recover_third_rt(cls, pts1, camera1, pts2, camera2, pts3):
        K, dist = Camera().K, Camera().distortion 
        undistorted_points_1 = cv2.undistortPoints(pts1, K, dist, R=camera1.R, P=camera1.get_projection_matrix())
        undistorted_points_2 = cv2.undistortPoints(pts2, K, dist, R=camera2.R, P=camera2.get_projection_matrix())

        points_3d_homog = cv2.triangulatePoints(
            camera1.get_projection_matrix(),
            camera2.get_projection_matrix(),
            undistorted_points_1,
            undistorted_points_2
        )
        points_3d = cv2.convertPointsFromHomogeneous(points_3d_homog.T)
                
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(points_3d, pts3, Camera().K, Camera().distortion) 
        return cv2.Rodrigues(rvecs)[0], tvecs, points_3d
    
    @classmethod
    def initial_triangulation(cls, image1, image2, image3):
        """ 1. Initial stereo camera reconstruction
            2. Third camera triangulation
        """
        sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)
        kp3, des3 = sift.detectAndCompute(image3, None)

        matches12 = flann.knnMatch(des1, des2, k=2)
        matches13 = flann.knnMatch(des1, des3, k=2)

        qidx12 = [m.queryIdx for m,n in list(filter(lambda x: x[0].distance < Frame.MATCHER_THREHSOLD*x[1].distance, matches12))]
        qidx13 = [m.queryIdx for m,n in list(filter(lambda x: x[0].distance < Frame.MATCHER_THREHSOLD*x[1].distance, matches13))]

        common_query_indexes = list(set(qidx12).intersection(qidx13))

        # match between 3 images
        good_12_matches = []
        for m, n in matches12:
            if m.queryIdx in common_query_indexes:
                good_12_matches.append(m)

        good_13_matches = []
        for m, n in matches13:
            if m.queryIdx in common_query_indexes:
                good_13_matches.append(m)

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good_12_matches]).reshape(-1,1,2)
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good_12_matches]).reshape(-1,1,2)
        pts3 = np.float64([kp3[m.trainIdx].pt for m in good_13_matches]).reshape(-1,1,2)
        
        R2, t2 = Scene.get_rt_from_essential(pts1, pts2)
            
        camera1 = Camera.create(np.eye(3, 3), np.zeros((3, 1)))
        camera2 = Camera.create(R2, t2)
        
        R3, t3, points_3d = Scene.recover_third_rt(camera1=camera1, camera2=camera2, pts1=pts1, pts2=pts2, pts3=pts3)

        frame1 = Frame(image1)
        frame1.R = camera1.R
        frame1.t = camera1.t
        frame1.points_3d = points_3d
        frame1.points_2d = pts1
        frame1.kp = kp1
        frame1.des = des1
        
        frame2 = Frame(image2)
        frame2.R = camera2.R
        frame2.t = camera2.t
        frame2.points_3d = points_3d
        frame2.points_2d = pts2
        frame2.kp = kp2
        frame2.des = des2
        
        frame3 = Frame(image3)
        frame3.R = R3
        frame3.t = t3
        frame3.points_3d = points_3d
        frame3.points_2d = pts3
        frame3.kp = kp3
        frame3.des = des3
        
        return frame1, frame2, frame3
    
    @classmethod
    def triangulation(cls, frame1, frame2, image3):
        """ 1. Stereo camera reconstruction
            2. Third camera triangulation
        """
        
        sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        kp1, des1 = sift.detectAndCompute(frame1.image, None)
        kp2, des2 = sift.detectAndCompute(frame2.image, None)
        kp3, des3 = sift.detectAndCompute(image3, None)

        matches12 = flann.knnMatch(des1, des2, k=2)
        matches13 = flann.knnMatch(des1, des3, k=2)

        qidx12 = [m.queryIdx for m,n in list(filter(lambda x: x[0].distance < Frame.MATCHER_THREHSOLD*x[1].distance, matches12))]
        qidx13 = [m.queryIdx for m,n in list(filter(lambda x: x[0].distance < Frame.MATCHER_THREHSOLD*x[1].distance, matches13))]

        common_query_indexes = list(set(qidx12).intersection(qidx13))

        # match between 3 images
        good_12_matches = []
        for m, n in matches12:
            if m.queryIdx in common_query_indexes:
                good_12_matches.append(m)

        good_13_matches = []
        for m, n in matches13:
            if m.queryIdx in common_query_indexes:
                good_13_matches.append(m)

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good_12_matches]).reshape(-1,1,2)
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good_12_matches]).reshape(-1,1,2)
        pts3 = np.float64([kp3[m.trainIdx].pt for m in good_13_matches]).reshape(-1,1,2)
                    
        camera1 = Camera.create(frame1.R, frame1.t)
        camera2 = Camera.create(frame2.R, frame2.t)
        
        R3, t3, points_3d = Scene.recover_third_rt(camera1=camera1, camera2=camera2, pts1=pts1, pts2=pts2, pts3=pts3)
        
        frame3 = Frame(image3)
        frame3.R = R3
        frame3.t = t3
        frame3.points_3d = points_3d
        frame3.points_2d = pts3
        frame3.kp = kp3
        frame3.des = des3
        
        return frame1, frame2, frame3