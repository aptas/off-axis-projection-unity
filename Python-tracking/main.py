# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:50:46 2019

@author: Santi
"""
import cv2
from PIL import Image
from webcam import Webcam
from detection import Detection
import numpy as np
from os.path import join
import asyncio
import json

async def tcp_echo_client(message, loop):
    # open connection with Unity 3D
    reader, writer = await asyncio.open_connection('127.0.0.1', 8080,
                                                   loop=loop)
    print('Send: %r' % message)

    # convert JSON to bytes
    message_json = json.dumps(message).encode()
    # send message
    writer.write(message_json)

    # wait for data from Unity 3D
    data = await reader.read(100)
    # we expect data to be JSON formatted
    data_json = json.loads(data.decode())
    print('Received:\n%r' % data_json)

    print('Close the socket')
    writer.close()
 
def show_image_with_data(frame, blinks, irises, err=None):
    """
    Helper function to draw points on eyes and display frame
    :param frame: image to draw on
    :param blinks: number of blinks
    :param irises: array of points with coordinates of irises
    :param err: for displaying current error in Lucas-Kanade tracker
    :return:
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if err:
        cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'blinks: ' + str(blinks), (10, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    for w, h in irises:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
    cv2.imshow('Eyeris detector', frame)


class ImageSource:
    """
    Returns frames from camera
    """
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_current_frame(self, gray=False):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        if not gray:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.capture.release()


class CascadeClassifier:
    """
    This classifier is trained by default in OpenCV
    """
    def __init__(self, glasses=True):
        if glasses:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye_tree_eyeglasses.xml'))
        else:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye.xml'))

    def get_irises_location(self, frame_gray):
        eyes = self.eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
        irises = []

        for (ex, ey, ew, eh) in eyes:
            iris_w = int(ex + float(ew / 2))
            iris_h = int(ey + float(eh / 2))
            irises.append([np.float32(iris_w), np.float32(iris_h)])
        print(irises)

        return np.array(irises)


class LucasKanadeTracker:
    """
    Lucaas-Kanade tracker used for minimizing cpu usage and blinks counter
    """
    def __init__(self, blink_threshold=9):
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.blink_threshold = blink_threshold

    def track(self, old_gray, gray, irises, blinks, blink_in_previous):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **self.lk_params)
        if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
            lost_track = True
            blink_in_previous = False
        elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in klt tracking
            lost_track = True
            if not blink_in_previous:
                blinks += 1
                blink_in_previous = True
        else:
            blink_in_previous = False
            irises = []
            for w, h in p1:
                irises.append([w, h])
            print(irises)
            irises = np.array(irises)
        return irises, blinks, blink_in_previous, lost_track


class EyerisDetector:
    """
    Main class which use image source, classifier and tracker to estimate iris postion
    Algorithm used in detector is designed for one person (with two eyes)
    It can detect more than two eyes, but it tracks only two
    """
    def __init__(self, image_source, classifier, tracker):
        self.tracker = tracker
        self.classifier = classifier
        self.image_source = image_source
        self.irises = []
        self.blink_in_previous = False
        self.blinks = 0

    def run(self):
        k = cv2.waitKey(30) & 0xff
        while k != 27:  # ESC
            frame = self.image_source.get_current_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                if lost_track:
                    self.irises = self.classifier.get_irises_location(gray)
            else:  # cannot track for some reason -> find irises
                self.irises = self.classifier.get_irises_location(gray)

            show_image_with_data(frame, self.blinks, self.irises)
            k = cv2.waitKey(30) & 0xff
            old_gray = gray.copy()

        self.image_source.release()
        cv2.destroyAllWindows()

class LegoTracker:
 
    def __init__(self, image_source, classifier, tracker):
        self.webcam = Webcam()
        self.webcam.start()        
        self.tracker = tracker
        self.classifier = classifier
        self.prev_irises = []
        self.irises = []
        self.blink_in_previous = False
        self.blinks = 0
          
        self.x_axis = 0.0
        self.y_axis = 0.0
        self.z_axis = 0.0
    
    def _update_image(self):
        # get image from webcam 
        image = self.webcam.get_current_frame()
        return self.irises
     
    def run(self):
        k = cv2.waitKey(30) & 0xff
        while k != 27:  # ESC
            self.prev_irises = self.irises
            frame = self.webcam.get_current_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                if lost_track:
                    self.irises = self.classifier.get_irises_location(gray)
            else:  # cannot track for some reason -> find irises
                self.irises = self.classifier.get_irises_location(gray)

            show_image_with_data(frame, self.blinks, self.irises)
            k = cv2.waitKey(30) & 0xff
            old_gray = gray.copy()
            self._eye_movement()

        self.webcam.release()
        cv2.destroyAllWindows()
 
    def _eye_movement(self):
        
        diff_x = 0
        diff_y = 0
        diff_z = 0
        curr_z = 0
        print(len(self.irises))
        print(self.irises)
        if(len(self.irises)>1 and len(self.prev_irises) > 1):
            prev_x = (self.prev_irises[0][0] + self.prev_irises[1][0]) / 2.
            prev_y = (self.prev_irises[0][1] + self.prev_irises[1][1]) / 2.
            prev_eye_dist = np.power(np.power(self.prev_irises[0][0] - self.prev_irises[1][0],2) +
                              np.power(self.prev_irises[0][1] - self.prev_irises[1][1],2),0.5)
            
            curr_x = (self.irises[0][0] + self.irises[1][0]) / 2.
            curr_y = (self.irises[0][1] + self.irises[1][1]) / 2.
            curr_eye_dist = np.power(np.power(self.irises[0][0] - self.irises[1][0],2) +
                              np.power(self.irises[0][1] - self.irises[1][1],2),0.5)
        
            diff_x = curr_x - prev_x
            diff_y = curr_y - prev_y
            diff_z = curr_eye_dist - prev_eye_dist
            
            message = {"eyeX": diff_x/100,
                       "eyeY": diff_y/100,
                       "eyeZ": curr_eye_dist/100}
            loop = asyncio.get_event_loop()
            loop.run_until_complete(tcp_echo_client(message, loop))
            #loop.close()
 
    def face_tracker(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Read the input image
        #img = cv2.imread('test.png')
        cap = cv2.VideoCapture(0)
        
        prev_face = []
        k = cv2.waitKey(30) & 0xff
        while k != 27:  # ESC
            _, img = cap.read()
        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if (len(prev_face) != 0 and len(faces) > 1):
                x_prev = prev_face[0][0]
                y_prev = prev_face[0][1]
                dists = np.zeros((len(faces),1))
                for k in range(len(faces)):
                    x = faces[k][0]
                    y = faces[k][1]
                    dists[k] = (x_prev-x)^2 + (y_prev-y)^2
                
                arg_min = np.argmin(dists)
                faces = [faces[arg_min]]
                    
                
        
            for (x, y , w ,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
            
            if(len(prev_face) != 0 and len(faces) != 0):
                diff_x = faces[0][0] - prev_face[0][0]
                diff_y = faces[0][1] - prev_face[0][1]
                diff_z = (faces[0][2] - prev_face[0][2] + faces[0][3] - prev_face[0][3]) / 2
                message = {"eyeX": diff_x/100,
                       "eyeY": diff_y/100,
                       "eyeZ": diff_z/100}
                loop = asyncio.get_event_loop()
                loop.run_until_complete(tcp_echo_client(message, loop))

            
            # Display the output
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prev_face = faces 
            k = cv2.waitKey(30) & 0xff

        cap.release()
    
    def main(self):
#        self.run()
        self.face_tracker()
        print('finished lego main')
 
# run instance of Lego Tracker 
legoTracker = LegoTracker(image_source=0, classifier=CascadeClassifier(glasses = False),
                                 tracker=LucasKanadeTracker())
legoTracker.main()