import cv2
import glob

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("video.mp4", fourcc, 20.0, (960, 540))

for path in glob.glob("intermediate/**.png"):
    frame = cv2.imread(path)
    out.write(frame) 
    cv2.waitKey(1)

out.release()
cv2.destroyAllWindows()