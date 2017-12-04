import cv2
import numpy as np

cap = cv2.VideoCapture('videos/checkerboard.mp4')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

if (cap.isOpened() == False):
    print("Error opening video stream or file")


counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, layers = frame.shape 
    frame = cv2.resize(frame, (int(width/2), int(height/2))) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    backup = frame.copy()
    # If found, add object points, image points (after refining them)
    if ret == True:
        ret2, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret2 == True:
            counter += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            #img = cv2.drawChessboardCorners(frame, (9,6), corners2, ret)
            height, width, layers =  frame.shape
            if counter > 600: break
            cv2.imwrite('images/img'+str(counter)+'.png', backup)
    else:
        break

cap.release()
cv2.destroyAllWindows()
