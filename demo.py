import cv2
import numpy as np

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        try:
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        except Exception as e:
            #print(r, (x0,y0), (x1,y1))
            pass


    return img1,img2

cv2.ocl.setUseOpenCL(False)

cap = cv2.VideoCapture('video.mp4')

old_frame = None

while(cap.isOpened()):
    ret, frame = cap.read()    
    if ret == True:
        height, width, layers =  frame.shape
        frame = cv2.resize(frame, (int(width/2), int(height/2))) 
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        if old_frame is None: 
            old_frame = frame
            continue

        # =============================
        # =============================
        # =============================

        
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(frame,None)
        kp2, des2 = orb.detectAndCompute(old_frame,None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)

        matches = sorted(matches, key = lambda x:x.distance)
        
        good = []
        pts1 = []
        pts2 = []
        for m in matches:
            if m.distance < 80:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)



        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
    
    
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 1,F)
        lines1 = lines1.reshape(-1,3)
        

        #img5,img6 = drawlines(frame,old_frame,lines1,pts1,pts2)


        img2 = cv2.drawKeypoints(frame, kp1, None,(255,0,0),4)



        cv2.imshow('image', img2)
        cv2.waitKey(1)


    else:
        break

    


    

cap.release()
cv2.destroyAllWindows()
