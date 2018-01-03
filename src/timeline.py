import cv2

import numpy as np 
import pandas as pd

from sklearn.neural_network import MLPRegressor

from src.camera import Camera
from src.utils.cube import generate_cube, draw_cube
from src.frame import Frame
from src.scene import Scene
from src.utils.functions import window

class Timeline(object):
    """ Main class that stores video """
    RESIZE_SCALE = 2
    KEYFRAME_DISTANCE = 30
    
    def __init__(self, path="videos/video13.mp4"):
        self.path_video = path # path to video
        self.buffer = [] # buffered video a.k.a images
        self.K = Camera().K # calibration matrix
        self.distortion = Camera().distortion # distortion coefs
        self.common_points = [] # common points between keyframes    
        
        self.__read_into_buffer()
        self.keyframes_indexes = [x for x in range(0, len(self.buffer), self.KEYFRAME_DISTANCE)]
        
        self.keyframes = []
        self.adjusted_keyframes = []
        self.__recover_keyframes()
        self.__adjust_keyframes()
        
        
    def __resize(self, frame):
        """ Resize video in hald """
        height, width, layers =  frame.shape
        frame = cv2.resize(frame, (int(width/2), int(height/2)))
        return frame
        
    def __read_into_buffer(self):
        """ Reads video into object buffer """
        cap = cv2.VideoCapture(self.path_video)
        buffer = []
        original_images_buffer = []
        while(cap.isOpened()):
            ret, frame = cap.read()    
            if ret == True:
                frame = self.__resize(frame)
                buffer.append(frame)
                original_images_buffer.append(frame)
            else:
                cap.release()
                
        self.buffer = buffer
                
    def __recover_keyframes(self):
        """  """
        keyframes = []
        
        frame1, frame2, frame3 = None, None, None
        
        for i, triple in enumerate(window(self.keyframes_indexes, 3)):
            image1, image2, image3 = self.buffer[triple[0]], self.buffer[triple[1]], self.buffer[triple[2]]
            if i > 0:
                frame1, frame2, frame3 = Scene.triangulation(frame2, frame3, image3)
                keyframes.append(frame3)
            else: # if initial
                frame1, frame2, frame3 = Scene.initial_triangulation(image1, image2, image3)
                keyframes = [frame1, frame2, frame3]
                
        self.keyframes = keyframes
                
    def __adjust_keyframes(self):
        self.adjusted_keyframes = [frame.bundle_adjustment() for frame in self.keyframes]
        
    
    def interpolate_frames_and_save(self):
        cube = generate_cube(1.0, [0,0,10])
        train = []
        for kf in self.adjusted_keyframes[:]:
            object_points = []
            for point in kf.create_camera_and_project(cube):
                x, y = point[0]
                object_points.append(x)
                object_points.append(y)

            train.append(object_points)

        df = pd.DataFrame(train)
        
        X = np.array([x*self.KEYFRAME_DISTANCE for x in range(df.shape[0])]).reshape(df.shape[0], 1)
        y = df
        
        mlp = MLPRegressor(
            max_iter=2000,
            learning_rate_init=0.01,
            random_state=42
        )
        mlp.fit(X, y)
        
        for i, image in enumerate(self.buffer):
            frame = Frame(image)
            res = mlp.predict(i)
            image = draw_cube(frame.image, res.reshape((-1, 1, 2)))

            cv2.imwrite("saved_images/{}.png".format("%03d" % i), image)
            
        
        