import cv2
import numpy as np

from src.main.triples import KeyFrame, KeyFrameTriple

class Timeline(object):
    __MATCHES_THRESHOLD = 200
    __MAX_VIEW_THRESHOLD = 100

    def __init__(self, path):
        """ Contructor
        
        Args:
            path (str): path to videofile 
        """
        self.video_file_path = path
        self.ratio = 2

    def __resize(self, frame):
        """ Resizes frame by ratio & returns new frame and grayscale image
        
        Args:
            frame: image

        Returns:
            frame: resized image
            gray: resized grayscale image
        """
        height, width, layers =  frame.shape
        frame = cv2.resize(frame, (int(width/self.ratio), int(height/self.ratio))) 
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        return frame, gray

    def __read_video_in_buffer(self, path):
        """ Reads video stream into buffer for future easy access
        
        Args:
            path (str): path to video file

        Returns:
            frames: list of video frames
            grayscale_frames : list of grayscale video frames
        """
        frames = []
        grayscale_frames = []

        cap = cv2.VideoCapture(self.video_file_path)
        
        while(cap.isOpened()):
            ret, frame = cap.read()    
            if ret == True:
                frame, gray = self.__resize(frame)
                frames.append(frame)
                grayscale_frames.append(gray)

            else:
                cap.release()

        cv2.destroyAllWindows() 

        return frames, grayscale_frames


    def construct_keyframes(self):
        """ Run video sequence and construct keyframes """
        cap = cv2.VideoCapture(self.video_file_path)

        print("Start reading video into buffer...")
        frames, grayscale_frames = self.__read_video_in_buffer(self.video_file_path)
        print("Video buffered...")

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        ftkf = frames[0] # first triple keyframe
        ftkf_index = 0
        counter = 0 # counter of number of frame
        view_counter = 0 # number of frames is looking in future
        triples = [] # keyframe triples

        for frame in frames:
            # compute matches
            kp1, des1 = orb.detectAndCompute(ftkf, None)
            kp2, des2 = orb.detectAndCompute(frame, None)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x: x.distance)
            good = matches[:int(len(matches) * 0.1)]

            # if to small matches found break and save kf triple
            if len(matches) < self.__MATCHES_THRESHOLD or view_counter > self.__MAX_VIEW_THRESHOLD:
                f1 = ftkf
                f2_index = counter - ftkf_index - int(view_counter/2)
                f2 = frames[f2_index]
                f3 = frame

                kf1 = KeyFrame(f1)
                kf2 = KeyFrame(f2)
                kf3 = KeyFrame(f3)

                triple = KeyFrameTriple(kf1, kf2, kf3, [])

                triples.append(triple)
                    
                ftkf = frames[-1]
                ftkf_index = f2_index
                view_counter = 0

            print(len(matches), ftkf_index, counter, view_counter, len(triples))


            # increment counters
            view_counter += 1
            counter += 1


        # while(cap.isOpened()):
        #     ret, frame = cap.read()    
        #     if ret == True:
        #         # preprocess image
        #         frame, gray = self.__resize(frame)
        #         if ftkf is None: 
        #             ftkf = frame
        #             ftkf_index = counter
        #             continue

        #         # compute matches
        #         kp1, des1 = orb.detectAndCompute(ftkf, None)
        #         kp2, des2 = orb.detectAndCompute(frame, None)

        #         matches = bf.match(des1,des2)
        #         matches = sorted(matches, key = lambda x: x.distance)
        #         good = matches[:int(len(matches) * 0.1)]

        #         # if to small matches found break and save kf triple
        #         if len(matches) < self.__MATCHES_THRESHOLD or view_counter > self.__MAX_VIEW_THRESHOLD:
        #             f1 = ftkf
        #             f2_index = counter - int(view_counter/2)
        #             f2 = frames[f2_index]
        #             f3 = frame

        #             kf1 = KeyFrame(f1)
        #             kf2 = KeyFrame(f2)
        #             kf3 = KeyFrame(f3)

        #             triple = KeyFrameTriple(kf1, kf2, kf3, [])

        #             triples.append(triple)
                    
        #             ftkf = frames[-1]
        #             ftkf_index = f2_index
        #             view_counter = 0

        #         print(len(matches), ftkf_index, counter, view_counter, len(triples))




        #         # cv2.imshow('image', frame)
        #         # cv2.waitKey(1)

        #         # add frame to buffer
        #         frames.append(frame)

        #     else:
        #         cap.release()

        #     view_counter += 1
        #     counter += 1

        # cv2.destroyAllWindows() 

        



    
