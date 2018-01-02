import numpy as np
import cv2

from src.utils.cube import generate_cube
from src.camera import Camera

first_frame_index = 0
second_frame_index = 25
thrid_frame_index = 50


def resize(frame):
    height, width, layers =  frame.shape
    frame = cv2.resize(frame, (int(width/2), int(height/2))) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray

cap = cv2.VideoCapture("videos/video5.mp4")

sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

buffer = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame, gray = resize(frame)

        buffer.append(frame)

    else:
        cap.release()

print("Buffered...")

kp1, des1 = sift.detectAndCompute(buffer[first_frame_index], None)
kp2, des2 = sift.detectAndCompute(buffer[second_frame_index], None)
kp3, des3 = sift.detectAndCompute(buffer[thrid_frame_index], None)


matches = flann.knnMatch(des1, des2, k=2)

good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)

pts1 = np.float64([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
pts2 = np.float64([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

c2 = Camera()

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
E = c2.K.T * F * c2.K
#E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
points, R, t, mask = cv2.recoverPose(E, pts1, pts2, pp=(486., 265.))

matches = flann.knnMatch(des1, des3, k=2)

good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append(m)

pts1 = np.float64([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
pts3 = np.float64([ kp3[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

F, mask = cv2.findFundamentalMat(pts1, pts3, cv2.FM_RANSAC)
E = c2.K.T * F * c2.K
#E, mask = cv2.findEssentialMat(pts1, pts3, focal=1.0, pp=(486., 265.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
_, R2, t2, _ = cv2.recoverPose(E, pts1, pts3, pp=(486., 265.))



camera1 = Camera()
camera1.R = np.eye(3, 3)
camera1.t = np.zeros((3, 1))

camera2 = Camera()
camera2.R = R
camera2.t = t

camera3 = Camera.create(R2, t2)




c = Camera()
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(c.K,  M_l)
P_r = np.dot(c.K,  M_r)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, pts1, pts2)
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_3d = point_4d[:3, :].T

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

upbound = [x*20 for x in initial]
lbound = [x*0.05 for x in initial]

bnds = [(x, y) for x, y in zip(upbound, lbound)]





result = diffev2(
        reprojection_distance, 
        x0=bnds, 
        args=(point_3d, points_2d),
        npop=10, 
        gtol=200,
        disp=False, 
        full_output=True, 
        itermon=mon, 
        maxiter=200
)


err = reprojection_distance(result[0], point_3d, points_2d)
print(err)




cube = generate_cube(1.0, [0,0,10])

image_points1 = camera1.project(cube)
image_points2 = camera2.project(cube)
image_points3 = camera3.project(cube)

for point in image_points1:
    x, y = point[0]
    cv2.circle(buffer[first_frame_index], (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    cv2.imshow("frame1", buffer[first_frame_index])

for point in image_points2:
    x, y = point[0]
    cv2.circle(buffer[second_frame_index], (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    cv2.imshow("frame2", buffer[second_frame_index])

for point in image_points3:
    x, y = point[0]
    cv2.circle(buffer[thrid_frame_index], (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    cv2.imshow("frame3", buffer[thrid_frame_index])

while True:
    cv2.waitKey(10)

cv2.destroyAllWindows() 
