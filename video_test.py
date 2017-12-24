import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)

from src.main.timeline import Timeline


# cap = cv2.VideoCapture('videos/video.mp4')

# old_frame = None
# first_frame_kps = None
# first_frame_desc = None
# prev_des = None

# orb = cv2.ORB_create()
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# while(cap.isOpened()):
#     ret, frame = cap.read()    
#     if ret == True:

#         # ========
#         # video preprocessing
#         # ========
#         height, width, layers =  frame.shape
#         frame = cv2.resize(frame, (int(width/2), int(height/2))) 
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         gray = np.float32(gray)
#         if old_frame is None: 
#             old_frame = frame
#             continue
#         # =========

#         # find key points of first frame
#         if not first_frame_kps:
#             kps, descs = orb.detectAndCompute(frame,None)
#             first_frame_kps, first_frame_desc = kps, descs
#             continue

#         kp, des = orb.detectAndCompute(frame, None)

#         matches = bf.match(first_frame_desc, des)
#         matches = sorted(matches, key = lambda x: x.distance)

#         print(query_discriptors)



#         prev_des = query_discriptors

#         # good = matches[:int(len(matches) * 0.1)]

#         # # select apropriate matches
#         # pts1 = []
#         # pts2 = []
#         # for m in matches[:int(len(matches) * 0.2)]:
#         #     #if m.distance < self.__DISTANCE:
#         #     pts2.append(kp2[m.trainIdx].pt)
#         #     pts1.append(kp1[m.queryIdx].pt)
#         # pts1 = np.int32(pts1)
#         # pts2 = np.int32(pts2)

        

#         cv2.imshow('image', frame)
#         cv2.waitKey(1)

# cap.release()
# cv2.destroyAllWindows()

timeline = Timeline("videos/video.mp4")
timeline.construct_keyframes()