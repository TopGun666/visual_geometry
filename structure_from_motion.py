import cv2
import numpy as np 

from src.camera import Camera
from src.utils.cube import generate_cube

cap = cv2.VideoCapture("videos/video6.mp4")

def resize(frame):
    height, width, layers =  frame.shape
    frame = cv2.resize(frame, (int(width/2), int(height/2))) 
    blured = blur = cv2.GaussianBlur(frame,(5,5),0)
    return frame

buffer = []

print("Buffering images...")
while(cap.isOpened()):
    ret, frame = cap.read()    
    if ret == True:
        frame = resize(frame)
        buffer.append(frame)
    else:
        cap.release()
print("Buffering done...")

def check_rotation_matrix(R):
    return abs(np.linalg.det(R)) - 1.0 < 1e-07 

class Feature(object):
    """ Image features """
    def __init__(self, points, keypoints, descriptors):
        self.points = points
        self.keypoints = keypoints
        self.descriptors = descriptors

def match_images(image_one, image_two):
    """ Match points between images

    Returns:
        f1, f2: features for image #1 and #2
    """
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image_one, None)
    kp2, des2 = orb.detectAndCompute(image_two, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)
    good = matches

    # sift = cv2.xfeatures2d.SIFT_create()
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # kp1, des1 = sift.detectAndCompute(image_one, None)
    # kp2, des2 = sift.detectAndCompute(image_two, None)

    # matches = flann.knnMatch(des1, des2, k=2)

    # good = []
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    pts1 = np.float64([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    return Feature(pts1, kp1, des1), Feature(pts2, kp2, des2)

def find_homography_inliers(image_one, image_two):
    features1, features2 = match_images(image_one, image_two)
    M, mask = cv2.findHomography(features1.points, features2.points, cv2.RANSAC, 5.0)
    return np.count_nonzero(mask), np.count_nonzero(mask) / len(mask)


min_pair, min_inliers_in_pair = None, 9999999.
# for i in range(len(buffer)):
#     inliers_count = find_homography_inliers(buffer[0], buffer[i])
#     print(i, inliers_count)
#     if inliers_count != 0 and min_inliers_in_pair > inliers_count:
#         min_pair = (0, i)
#         min_inliers_in_pair = inliers_count

# for i in range(len(buffer) - 1):
#     print("Intermediate result:", min_pair, min_inliers_in_pair)
#     for j in range(i + 1, len(buffer)):
#         _, ratio = find_homography_inliers(buffer[i], buffer[j])
#         print(i, j, ratio)
#         if min_inliers_in_pair > ratio and ratio > 0.0:
#             min_pair = (i, j)
#             min_inliers_in_pair = ratio

# print("Result:", min_pair, min_inliers_in_pair)




### pair with smallest count of inliers
img1, img2 = buffer[0], buffer[55]

features_1, features_2 = match_images(img1, img2)

E, mask = cv2.findEssentialMat(
    features_1.points, 
    features_2.points, 
    focal=1.0, 
    pp=(486.2, 265.59), 
    method=cv2.RANSAC, 
    prob=0.999, 
    threshold=1.0
)


points, R, t, mask = cv2.recoverPose(
    E, 
    features_1.points, 
    features_2.points, 
    focal=1.0,
    pp=(486.2, 265.59),
    mask=mask
)


K, dist = Camera().K, Camera().distortion 

camera1 = Camera.create(np.eye(3, 3), np.zeros((3, 1)))
camera2 = Camera.create(R, t)

undistorted_points_1 = cv2.undistortPoints(features_1.points, K, dist, R=np.eye(3, 3), P=camera1.get_projection_matrix())
undistorted_points_2 = cv2.undistortPoints(features_2.points, K, dist, R=R, P=camera2.get_projection_matrix())

points_3d_homog = cv2.triangulatePoints(
    camera1.get_projection_matrix(),
    camera2.get_projection_matrix(),
    undistorted_points_1,
    undistorted_points_2
)
points_3d = cv2.convertPointsFromHomogeneous(points_3d_homog.T)



for point in camera1.project(points_3d):
    try:
        x, y = point[0]
        cv2.circle(img1, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    except Exception as e:
        print(e)
    
for point in features_1.points:
    try:
        x, y = point[0]
        cv2.circle(img1, (int(x), int(y)), 2, (0, 0, 255), thickness=2, lineType=8, shift=0)
    except Exception as e:
        print(e)

while True:
    cv2.imshow("img1", img1)
    #cv2.imshow("img2", img2)
    cv2.waitKey(10)


cv2.destroyAllWindows() 