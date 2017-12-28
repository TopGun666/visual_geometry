import cv2
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from scipy.spatial.distance import euclidean

from src.camera import Camera

def reprojection_distance(rt, points_3d, points_2d):
    Rt = rt.reshape((3, 4))
    R, t, _ = np.hsplit(Rt, np.array((3, 4)))
    t = t.reshape((1, 3))

    camera = Camera()

    projected = cv2.projectPoints(
        points_3d,
        R,
        t,
        camera.K,
        camera.distortion
    )[0]

    d = euclidean(projected.reshape(projected.shape[0]*2), points_2d.reshape(points_2d.shape[0]*2))

    return d

class Frame(object):
    def __init__(self, image):
        self.image = image
        self.R = None
        self.t = None
        self.points_3d = None
        self.interval_descriptors = None
        self.interval_keypoints = None

    def __str__(self):
        return "<Frame:\nimage:{}\nR:{}\nt:{}\n3d_points:{}\n>".format(self.image.shape, self.R, self.t, self.points_3d)


        #np.hstack((np.eye(3, 3), np.zeros((3, 1))))


    def optimize(self):
        # create feature extractor and feature matcher
        sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # find matches between interval keypoints and intermediate frame keypoints
        keypoints, descriptors = sift.detectAndCompute(self.image, None)
        matches = flann.knnMatch(self.interval_descriptors, descriptors, k=2)
        
        # points of intermediate frames
        points_2d = np.float64([ keypoints[m.trainIdx].pt for m, _ in matches ])

        # TODO: optimize d(cameraProject(points_3d), points_2d)
        initial = np.ones(12)

        print(self.points_3d.shape, points_2d.shape)

        optimized = minimize(
            reprojection_distance,
            initial, 
            (self.points_3d, points_2d),
            method='L-BFGS-B'
        )

        R, t = np.hsplit(optimized.x.reshape((3, 4)), np.array((3, 4)))[0], np.hsplit(optimized.x.reshape((3, 4)), np.array((3, 4)))[1].reshape((1, 3))

        #print(R, t)

        # camera = Camera()
        # ret, rvecs, tvecs = cv2.solvePnP(self.points_3d, points_2d, camera.K, camera.distortion)

        self.R = R
        self.t = t

        print("Frame optimized...")

        return self