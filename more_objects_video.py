import cv2
import numpy as np 

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


from scipy.optimize import minimize, rosen, rosen_der, basinhopping, fmin, fmin_cg, differential_evolution
from scipy.spatial.distance import euclidean, cdist

from src.camera import Camera
from src.utils.cube import generate_cube
from dlt import DLTcalib, project

# mystic
from mystic.penalty import quadratic_inequality
from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor


mon = VerboseMonitor(10)


class Frame(object):
    def __init__(self, image, ad):
        self.image = image
        self.camera = None
        self.assoc_dict = ad

    def get_common_points(self, indexes):
        res = []
        for idx in self.assoc_dict.keys():
            if idx in indexes:
                res.append(self.assoc_dict[idx])

        return res

def reprojection_distance(input, points_3d, points_2d):
    r1, r2, r3, t1, t2, t3 = input[0], input[1], input[2], input[3], input[4], input[5]

    R = cv2.Rodrigues(np.array([r1, r2, r3]))[0]

    t = np.array([[t1], [t2], [t3]])

    # Rt = input.reshape((3, 4))
    # R, t, _ = np.hsplit(Rt, np.array((3, 4)))
    # t = t.reshape((1, 3))

    camera = Camera()

    projected = cv2.projectPoints(
        points_3d,
        R,
        t,
        camera.K,
        camera.distortion
    )[0]

    vec1 = projected.reshape(projected.shape[0]*2)
    vec2 = points_2d.reshape(points_2d.shape[0]*2)

    res = 0
    for v1, v2 in zip(vec1, vec2):
        res += pow(v1 - v2, 2)


    # print(list(vec1))
    # print(list(vec2))
    # print(res)

    d = euclidean(projected.reshape(projected.shape[0]*2), points_2d.reshape(points_2d.shape[0]*2))

    d2 = cdist(
        np.array([projected.reshape(projected.shape[0]*2)]), 
        np.array([points_2d.reshape(points_2d.shape[0]*2)]), 
        'sqeuclidean'
    )

    # print(d2)
    # print(d)

    return res


cap = cv2.VideoCapture("videos/video6.mp4")

def resize(frame):
    height, width, layers =  frame.shape
    frame = cv2.resize(frame, (int(width/2), int(height/2))) 
    return frame

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

initial_frame = []
index_intersection = []
kp1, des1 = [], []
global_matches = []
frames = []

while(cap.isOpened()):
    ret, frame = cap.read()    
    if ret == True:
        frame = resize(frame)

        # get pattern image
        if len(initial_frame) == 0:
            initial_frame = frame
            kp1, des1 = sift.detectAndCompute(initial_frame, None)
            continue

        # create matcher


        # find the keypoints and descriptors with SIFT
        kp2, des2 = sift.detectAndCompute(frame, None)
        

        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]

    
        query_indexes = []
        mtch = []

        assoc_dict = {}

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                query_indexes.append(m.queryIdx)

                if m.queryIdx in index_intersection:
                    matchesMask[i]=[1,0]
                    mtch.append(m)

                    assoc_dict[m.queryIdx] = kp2[m.trainIdx].pt

        frames.append(Frame(frame, assoc_dict))

        if len(index_intersection) == 0:
            index_intersection = query_indexes
        else:
            index_intersection = list(set(index_intersection).intersection(query_indexes))

        print(len(index_intersection))

        global_matches = mtch
        if (len(index_intersection) < 20): cap.release()

        # draw_params = dict(matchColor = (0,255,0),
        #                 singlePointColor = (255,0,0),
        #                 matchesMask = matchesMask,
        #                 flags = 0)

        # image_to_draw = cv2.drawMatchesKnn(initial_frame, kp1, frame, kp2, matches, None, **draw_params)

        # cv2.imshow("img", image_to_draw)
        # cv2.waitKey(100)


    else:
        cap.release()



pts1 = np.float64([ kp1[m.queryIdx].pt for m in global_matches ]).reshape(-1,1,2)
pts2 = np.float64([ kp2[m.trainIdx].pt for m in global_matches ]).reshape(-1,1,2)


#c2 = Camera()
#F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
#E = c2.K.T * F * c2.K
E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
points, R, t, mask = cv2.recoverPose(E, pts1, pts2, pp=(486., 265.))

# homograpy
c = Camera()
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(c.K,  M_l)
P_r = np.dot(c.K,  M_r)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, pts1, pts2)
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T

# print(point_3d)
# print(index_intersection)
# print(len(index_intersection))
# print(frames[6].get_common_points(index_intersection))

tmp_rod = cv2.Rodrigues(R)[0]
tmp_tvec = t
initial = [
    tmp_rod[0],
    tmp_rod[1],
    tmp_rod[2],
    tmp_tvec[0],
    tmp_tvec[1],
    tmp_tvec[2]  
]
initial = [x[0] for x in initial]

upbound = [x*1.1 for x in initial]
lbound = [x*0.9 for x in initial]

bnds = [(x, y) for x, y in zip(upbound, lbound)]



for saved_frame in frames:
    points_2d = np.array(saved_frame.get_common_points(index_intersection))
    if len(points_2d) == 0: continue

    # initial = list(np.ones(6))
    # optimized = minimize(
    #     reprojection_distance,
    #     initial, 
    #     args=(point_3d, points_2d),
    #     method='SLSQP'
    # )

    # print(optimized)
    # print(list(optimized.x))
    # print(initial)
    # print(reprojection_distance(optimized.x, point_3d, points_2d))


    # result = diffev2(
    #     reprojection_distance, 
    #     x0=bnds, 
    #     args=(point_3d, points_2d),
    #     npop=10, 
    #     gtol=200,
    #     disp=False, 
    #     full_output=True, 
    #     itermon=mon, 
    #     maxiter=200
    # )

    # #print(result[0])

    # err = reprojection_distance(result[0], point_3d, points_2d)
    # print(err)

    c2 = Camera()
    _, R_recovered, t_recovered, _ = cv2.solvePnPRansac(point_3d, points_2d, c2.K, c2.distortion)
    R_recovered = cv2.Rodrigues(R_recovered)[0]

    # qwe = result[0] #optimized.x
    # r1, r2, r3, t1, t2, t3 = qwe[0], qwe[1], qwe[2], qwe[3], qwe[4], qwe[5]

    # R_recovered = cv2.Rodrigues(np.array([r1, r2, r3]))[0]
    # t_recovered = np.array([[t1], [t2], [t3]])


    recovered_camera = Camera.create(R_recovered, t_recovered)

    cube = generate_cube(1.0, [0,0,10])
    image_points = recovered_camera.project(cube)

    for point in image_points:
        x, y = point[0]
        try:
            print(x, y)
            cv2.circle(saved_frame.image, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
            cv2.imshow("recovered", saved_frame.image)
        except Exception as e:
            print(e)

    cv2.waitKey(1000)
































# points_2d = np.array(frames[10].get_common_points(index_intersection))
# result = diffev2(
#         reprojection_distance, 
#         x0=bnds, 
#         args=(point_3d, points_2d),
#         npop=10, 
#         gtol=200,
#         disp=False, 
#         full_output=True, 
#         itermon=mon, 
#         maxiter=300
# )

# print(result)
# err1 = reprojection_distance(initial, point_3d, points_2d)
# err2 = reprojection_distance(result[0], point_3d, points_2d)
# print(err1, err2)

# qwe = result[0] #optimized.x
# r1, r2, r3, t1, t2, t3 = qwe[0], qwe[1], qwe[2], qwe[3], qwe[4], qwe[5]

# R_recovered = cv2.Rodrigues(np.array([r1, r2, r3]))[0]
# t_recovered = np.array([[t1], [t2], [t3]])

























# camera1 = Camera()
# camera1.R = np.eye(3, 3)
# camera1.t = np.zeros((3, 1))

# camera2 = Camera()
# camera2.R = R
# camera2.t = t

# camera_recovered = Camera.create(R_recovered, t_recovered)

# cube = generate_cube(1.0, [0,0,10])

# image_points1 = camera1.project(cube)
# image_points2 = camera2.project(cube)
# image_points3 = camera_recovered.project(cube)

# for point in image_points1:
#     x, y = point[0]
#     cv2.circle(initial_frame, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
#     cv2.imshow("frame1", initial_frame)

# for point in image_points2:
#     x, y = point[0]
#     cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
#     cv2.imshow("frame2", frame)

# for point in image_points3:
#     x, y = point[0]
#     cv2.circle(frames[10].image, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
#     cv2.imshow("recovered", frames[10].image)




# while True:
#     cv2.waitKey(10)


cv2.destroyAllWindows() 
