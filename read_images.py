import cv2
import glob

for path in glob.glob("saved_images/**.png"):
    img = cv2.imread(path)

    cv2.imshow("frame", img)
    cv2.waitKey(500)
    #print(path)