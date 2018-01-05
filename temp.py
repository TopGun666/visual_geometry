import cv2
import glob
   
for i, path in enumerate(glob.glob("intermediate/**.png")):
    print(i)
    img = cv2.imread(path)
    cv2.imshow("frame", img)
    cv2.waitKey(30)