import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

from src.main.timeline import Timeline
from src.camera import Camera
from src.utils.cube import generate_cube
from src.main.frame import reprojection_distance





timeline = Timeline("videos/video.mp4")
timeline.compute_keyframe_triples()
_, frames = timeline.recover_keyframes()
timeline.fill_intermediate_frames()

#timeline.video_frames[5].optimize()

cube = generate_cube(1.0, [0,0,10])

for f in timeline.video_frames:
    camera = Camera.create(f.R, f.t)
    image_points = camera.project(cube)

    print(image_points)

#     # for point in image_points:
#     #     x, y = point[0]
#     #     cv2.circle(f.image, (int(x), int(y)), 3, (255, 0, 0), thickness=2, lineType=8, shift=0)
#     #     cv2.imshow("frame", f.image)

#     # cv2.waitKey(10)




