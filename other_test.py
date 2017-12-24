import cv2
import numpy as np

from src.utils.cube import generate_cube
from src.camera import Camera

cv2.ocl.setUseOpenCL(False)


img1 = cv2.imread("img10.png")
img2 = cv2.imread("img60.png")
img3 = cv2.imread("img200.png")


initial_camera = Camera()
initial_camera.R = np.eye(3, 3)
initial_camera.t = np.array([0.0, 0.0, 0.0])

camera = Camera()
camera.compute_camera_extrinsic(img1, img2)

camera3 = Camera()

triangulation = camera.triangulation






sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

gkp1, gdes1 = [ kp1[m.queryIdx] for m in good ], [ des1[m.queryIdx] for m in good ]
gkp2, gdes2 = [ kp2[m.trainIdx] for m in good ], [ des2[m.trainIdx] for m in good ]

hash_map_12 = dict([(m.trainIdx, m.queryIdx) for m in good])

matches23 = flann.knnMatch(np.asarray(gdes2,np.float32),np.asarray(des3,np.float32), k=2)

good23 = []
for m,n in matches23:
    if m.distance < 0.7*n.distance:
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


pts1 =  np.float64([ kp1[index].pt for index in indexes1 ]).reshape(-1,1,2)
pts2 =  np.float64([ kp2[index].pt for index in indexes2 ]).reshape(-1,1,2)
pts3 =  np.float64([ kp3[index].pt for index in indexes3 ]).reshape(-1,1,2)


F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
E, mask = cv2.findEssentialMat(pts1, pts2, focal=0.25, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
points, R, t, mask = cv2.recoverPose(E, pts1, pts2, pp=(486., 265.))

camera.R = R
camera.t = t


M_r = np.hstack((camera.R, camera.t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(camera.K,  M_l)
P_r = np.dot(camera.K,  M_r)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, pts1, pts2)
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T
        


ret, rvecs, tvecs = cv2.solvePnP(point_3d, pts3, camera.K, camera.distortion)

camera3.R = rvecs
camera3.t = tvecs




cube = generate_cube(1.0, [0,0,10])

image_points = initial_camera.project(cube)
image_points2 = camera.project(cube)
image_points3 = camera3.project(cube)


for point in image_points2:
    x, y = point[0]
    cv2.circle(img2, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #print("1>>>", x, y)

for point in image_points:
    x, y = point[0]
    cv2.circle(img1, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #print("1>>>", x, y)


for point in image_points3:
    x, y = point[0]
    cv2.circle(img3, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    print("3>>>", x, y)

while True:
    cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.imshow("img3",img3)
    cv2.waitKey(1)


cv2.destroyAllWindows()