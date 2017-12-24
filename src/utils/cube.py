import numpy as np

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