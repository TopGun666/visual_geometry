import cv2
cv2.ocl.setUseOpenCL(False)

import numpy as np

from src.camera import Camera



img1 = cv2.imread("img10.png")
img2 = cv2.imread("img20.png")

height1, width1, _ = img1.shape 
height2, width2, _ = img2.shape 

#img1 = cv2.resize(img1, (int(width1/2), int(height1/2))) 
#img2 = cv2.resize(img2, (int(width2/2), int(height2/2))) 


initial_camera = Camera()
initial_camera.R = np.eye(3, 3)
initial_camera.t = np.array([0.0, 0.0, 0.0])

camera = Camera()
camera.compute_camera_extrinsic(img1, img2)


def generate_cube(scale=1, shifting=[0, 0, 0]):
    """ Generates cube in homogenious coordinates """

    world_coords = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1 ], [1, 1, 1], [-1, 1, 1]
    ]

    res = []
    for point in world_coords:
        shifted = [a+b for a, b in zip(point, shifting)]
        scaled = [axis * scale for axis in shifted]
        #scaled.append(1.0)
        res.append(np.array(scaled))

    return np.array(res)



# for point in generate_cube(0.013, [25, 25, 0]):
#     x, y, _ = initial_camera.project(point)
#     x2, y2, _ = camera.project(point)
#     cv2.circle(img1, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
#     cv2.circle(img2, (int(x2), int(y2)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)

#     print(point)
#     print("1>>>", x, y)
#     print("2>>>", x2, y2)

cube = generate_cube(1.0, [0,0,10])

image_points = initial_camera.project(cube)
image_points2 = camera.project(cube)

for point in image_points2:
    x, y = point[0]
    cv2.circle(img2, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #print("1>>>", x, y)

for point in image_points:
    x, y = point[0]
    cv2.circle(img1, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #print("1>>>", x, y)

while True:
    cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.waitKey(1)


cv2.destroyAllWindows()

