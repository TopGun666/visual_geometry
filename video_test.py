import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

from src.main.timeline import Timeline


timeline = Timeline("videos/video.mp4")
timeline.compute_keyframe_triples()
#timeline.recover_keyframes_cameras()

first_triple = timeline.keyframe_triples[0]

kf1, kf2, kf3 = first_triple.recover_cameras()

print(kf1, kf2, kf3)

