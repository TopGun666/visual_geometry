import numpy as np
import cv2

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


def draw_cube(img, points):
    """ Draw lines to form a cube """
    points = points.astype(int)
    points = [x[0] for x in points]
    x, y = [point[0] for point in points], [point[1] for point in points]
    
    cv2.line(img, (x[0], y[0]), (x[1], y[1]), (255,255,0), 2)
    cv2.line(img, (x[1], y[1]), (x[2], y[2]), (255,255,0), 2)
    cv2.line(img, (x[2], y[2]), (x[3], y[3]), (255,255,0), 2)
    cv2.line(img, (x[3], y[3]), (x[0], y[0]), (255,255,0), 2)
    
    cv2.line(img, (x[4], y[4]), (x[5], y[5]), (255,255,0), 2)
    cv2.line(img, (x[5], y[5]), (x[6], y[6]), (255,255,0), 2)
    cv2.line(img, (x[6], y[6]), (x[7], y[7]), (255,255,0), 2)
    cv2.line(img, (x[7], y[7]), (x[4], y[4]), (255,255,0), 2)
    
    cv2.line(img, (x[0], y[0]), (x[4], y[4]), (255,255,0), 2)
    cv2.line(img, (x[1], y[1]), (x[5], y[5]), (255,255,0), 2)
    cv2.line(img, (x[2], y[2]), (x[6], y[6]), (255,255,0), 2)
    cv2.line(img, (x[3], y[3]), (x[7], y[7]), (255,255,0), 2)

    return img