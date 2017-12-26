import cv2
import numpy as np

from src.camera import Camera

class KeyFrame(object):
    """ Object for keyframe parameters """

    def __init__(self, frame):
        """ Constructor
        
        Args:
            frame: 2d numpy matrix with image
        """
        self.frame = frame
        self.R = None
        self.t = None

class KeyFrameTriple(object):
    """ Object for keyframe triples logic """
    
    def __init__(
        self, 
        f1, 
        f2, 
        f3
    ):
        """ Constructor

        Args:
            f1 (KeyFrame): key frame #1
            f2 (KeyFrame): key frame #2
            f3 (KeyFrame): key frame #3
            reconstruction_3d: points that forms 3d reconstruction
        """
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.reconstruction_3d = None

    
    def __initial_stereo_reconstruction(self):
        """ Initial stereo reconstruction
        Performed by computing essential matrix and recovering R and t for first and second camera.
        Than reconstruct 3d scene and recover third matrix from 3d point cloud.
        """
        print("Initial stereo reconstruction started...")

        # fixind 1st camera
        self.f1.R =  np.eye(3, 3)
        self.f1.t = np.array([0.0, 0.0, 0.0])

        # create feature extractor and feature matcher
        sift = cv2.xfeatures2d.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # extract features and descriptors
        kp1, des1 = sift.detectAndCompute(self.f1.frame, None)
        kp2, des2 = sift.detectAndCompute(self.f2.frame, None)
        kp3, des3 = sift.detectAndCompute(self.f3.frame, None)

        # match features and select good ones
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)

        gkp1, gdes1 = [ kp1[m.queryIdx] for m in good ], [ des1[m.queryIdx] for m in good ]
        gkp2, gdes2 = [ kp2[m.trainIdx] for m in good ], [ des2[m.trainIdx] for m in good ]

        # create hashmap to latter match features between three images
        hash_map_12 = dict([(m.trainIdx, m.queryIdx) for m in good])

        matches23 = flann.knnMatch(np.asarray(gdes2, np.float32),np.asarray(des3, np.float32), k=2)

        good23 = []
        for m,n in matches23:
            if m.distance < 0.8*n.distance:
                good23.append(m)

        gkp2, gdes2 = [ kp3[m.trainIdx] for m in good23 ], [ des3[m.trainIdx] for m in good23 ]
        hash_map_23 = dict([(m.trainIdx, m.queryIdx) for m in good23])

        indexes1 = []
        indexes2 = []
        indexes3 = []
        for key in hash_map_23.keys():
            if hash_map_23[key] in hash_map_12.keys():
                indexes1.append(hash_map_12[hash_map_23[key]])
                indexes2.append(hash_map_23[key])
                indexes3.append(key)
        
        # print(">>>>>>", len(indexes1))
        # print([ kp1[index].pt for index in indexes1 ])
        # print([ kp2[index].pt for index in indexes2 ])
        # print([ kp3[index].pt for index in indexes3 ])

        # get points in all three images
        pts1 =  np.float64([ kp1[index].pt for index in indexes1 ]).reshape(-1,1,2)
        pts2 =  np.float64([ kp2[index].pt for index in indexes2 ]).reshape(-1,1,2)
        pts3 =  np.float64([ kp3[index].pt for index in indexes3 ]).reshape(-1,1,2)

        # compute F and E for second camera
        # recover R and t from E
        # TODO: fix focal length and principal point
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=0.25, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        points, R, t, mask = cv2.recoverPose(E, pts1, pts2, pp=(486., 265.))

        self.f2.R = R
        self.f2.t = t

        camera = Camera() # NOTE: this is only for calibration and distortion

        # compute projection matrixes for camera #1 and #2
        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1)))) #np.hstack((self.f1.R, self.f1.t))
        M_r = np.hstack((self.f2.R, self.f2.t))
        P_l = np.dot(camera.K,  M_l)
        P_r = np.dot(camera.K,  M_r)

        # triangulate points and get 3d reconstruction
        point_4d_hom = cv2.triangulatePoints(P_l, P_r, pts1, pts2)
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        # compute R and t for camera #3
        ret, rvecs, tvecs = cv2.solvePnP(point_3d, pts3, camera.K, camera.distortion)

        self.f3.R = rvecs
        self.f3.t = tvecs

        print("Initial stereo reconstruction finished...")
        return self.f1, self.f2, self.f3

    def recover_cameras(self):
        if not self.f1.R and not self.f1.t and not self.f2.R and not self.f2.t and not self.reconstruction_3d:
            # case of initial camera recovering via fundamential matrix
            return self.__initial_stereo_reconstruction()

        elif self.f1.R and self.f1.t and self.f2.R and self.f2.t:
            # case when we need to recover 3rd camera via first two
            pass
        else:
            raise Exception("Unknown behavior in Keyframe triple recovering...")





    