"""
NOTE: run it from root or refine coresponding pathes
"""

import cv2
import glob
import numpy as np

def get_images(path = "resources/calibration_images/"):
    pattern = path + "**.png"

    images = []
    for img_path in glob.glob(pattern):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    return images


def get_and_save_calibrated_matrix():
    images = get_images()

    if len(images) == 0:
        raise EnvironmentError

    # for checkerboard pattern detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # arrays for 3d space points & image points
    objpoints = []
    imgpoints = []

    for img in images:
        print(".")
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("CALIBRATED MATRIX\n", mtx)

    np.save("resources/camera_matrix", mtx)
    np.save("resources/distortion", dist)
    np.save("resources/rvecs", rvecs)
    np.save("resources/tvecs", tvecs)  

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    get_and_save_calibrated_matrix()
