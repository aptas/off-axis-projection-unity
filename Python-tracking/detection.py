# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:50:00 2019

@author: Santi
"""

import cv2
  
class Detection(object):
  
    def get_items_in_image(self, item_cascade_path, image):
   
        # detect items in image
        item_cascade = cv2.CascadeClassifier(item_cascade_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        items = item_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=36)
         
        # debug: draw rectangle around detected items 
        for (x,y,w,h) in items:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
 
        # debug: show detected items in image
        cv2.imshow('OpenCV Detection', image)
        cv2.waitKey(100)
             
        # return items
        return items