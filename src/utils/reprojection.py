import cv2
import numpy as np 

from src.camera import Camera

def reprojection_distance(input, points_3d, points_2d):
    r1, r2, r3, t1, t2, t3 = input[0], input[1], input[2], input[3], input[4], input[5]
    R = cv2.Rodrigues(np.array([r1, r2, r3]))[0]
    t = np.array([[t1], [t2], [t3]])
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

    return res