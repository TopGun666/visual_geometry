import cv2
import numpy as np

from src.camera import Camera

cv2.ocl.setUseOpenCL(False)

cap = cv2.VideoCapture('videos/video.mp4')

old_frame = None

camera = Camera()

counter = 0
while(cap.isOpened()):
    counter += 1

    ret, frame = cap.read()    
    if ret == True:
        height, width, layers =  frame.shape
        frame = cv2.resize(frame, (int(width/2), int(height/2))) 
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        if old_frame is None: 
            old_frame = frame
            continue

        # F = camera.compute_fundamential_matrix(frame, old_frame)
        # print(F)

        cv2.imshow('image', frame)
        cv2.waitKey(1)

        if counter == 10 or counter == 200:
            cv2.imwrite('img'+str(counter)+'.png', frame)

        print(counter)

cap.release()
cv2.destroyAllWindows()
