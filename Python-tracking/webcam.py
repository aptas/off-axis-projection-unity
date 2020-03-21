# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:49:59 2019

@author: Santi
"""

import cv2
from threading import Thread
   
class Webcam:
   
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]
        self.started = False
           
    # create thread for capturing images
    def start(self):
        self.started = True
        Thread(target=self._update_frame, args=()).start()
   
    def _update_frame(self):
        while(self.started):
            self.current_frame = self.video_capture.read()[1]
                   
    # get the current frame
    def get_current_frame(self):
        return self.current_frame
    def release(self):
        self.video_capture.release()
        self.started = False