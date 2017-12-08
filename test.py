import cv2
import numpy as np

from src.camera import Camera

cv2.ocl.setUseOpenCL(False)

img1 = cv2.imread("img10.png")
img2 = cv2.imread("img200.png")

height1, width1, _ = img1.shape 
height2, width2, _ = img2.shape 

img1 = cv2.resize(img1, (int(width1/2), int(height1/2))) 
img2 = cv2.resize(img2, (int(width2/2), int(height2/2))) 




initial_camera = Camera()
initial_camera.R = np.diag(np.ones(3))
initial_camera.t = np.zeros([3, 1])

camera = Camera()
camera.compute_camera_extrinsic(img1, img2)


def generate_cube(scale=1):
    """ Generates cube in homogenious coordinates """

    world_coords = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1 ], [1, 1, 1], [-1, 1, 1]
    ]

    res = []
    for point in world_coords:
        scaled = [axis * scale for axis in point]
        scaled.append(1)
        res.append(np.array(scaled))

    return res



for point in generate_cube(0.1):
    x, y, _ = initial_camera.project(point)
    #x2, y2, _ = camera.project(point)
    cv2.circle(img1, (int(x + img1.shape[0]/2), int(y  + img1.shape[1] / 2)), 5, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #cv2.circle(img2, (int(x2 + img2.shape[0]/2), int(y2  + img2.shape[1] / 2)), 5, (255, 0, 0), thickness=2, lineType=8, shift=0)


while True:
    cv2.imshow("img1",img1)
    #cv2.imshow("img2",img2)
    cv2.waitKey(1)


cv2.destroyAllWindows()

