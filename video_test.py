import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

from src.main.timeline import Timeline
from src.camera import Camera
from src.utils.cube import generate_cube


timeline = Timeline("videos/video.mp4")
timeline.compute_keyframe_triples()
cameras, frames = timeline.recover_keyframes_cameras()


cube = generate_cube(1.0, [0,0,10])

for i, (camera, frame) in enumerate(zip(cameras, frames)):
    image_points = camera.project(cube)
    for point in image_points:
        x, y = point[0]
        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
        cv2.imshow(str(i), frame)

while True:
    cv2.waitKey(1)
