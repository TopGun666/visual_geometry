import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

from src.main.timeline import Timeline
from src.camera import Camera
from src.utils.cube import generate_cube


timeline = Timeline("videos/video.mp4")
timeline.compute_keyframe_triples()
#timeline.recover_keyframes_cameras()

first_triple = timeline.keyframe_triples[0]

kf1, kf2, kf3 = first_triple.recover_cameras()

camera1 = Camera.create(kf1.R, kf1.t)
camera2 = Camera.create(kf2.R, kf2.t)
camera3 = Camera.create(kf3.R, kf3.t)

cube = generate_cube(1.0, [0,0,10])

image_points = camera1.project(cube)
image_points2 = camera2.project(cube)
image_points3 = camera3.project(cube)


for point in image_points2:
    x, y = point[0]
    cv2.circle(kf1.frame, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #print("1>>>", x, y)

for point in image_points:
    x, y = point[0]
    cv2.circle(kf2.frame, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    #print("1>>>", x, y)


for point in image_points3:
    x, y = point[0]
    cv2.circle(kf3.frame, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
    print("3>>>", x, y)

while True:
    cv2.imshow("img1",kf1.frame)
    cv2.imshow("img2",kf2.frame)
    cv2.imshow("img3",kf3.frame)
    cv2.waitKey(1)