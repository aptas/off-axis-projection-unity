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
    #print('Received:\n%r' % data_json)

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

class CascadeClassifier:
    """
    This classifier is trained by default in OpenCV
    """
    def __init__(self, detect_faces=True, glasses=False):
        if glasses:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye_tree_eyeglasses.xml'))
        else:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye.xml'))
        if detect_faces:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.prev_face = []
            self.sp = []
            self.sp2 = []
            self.sl = 0
            self.smooth_init = False

    def get_irises_location(self, frame_gray):
        eyes = self.eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
        irises = []

        for (ex, ey, ew, eh) in eyes:
            iris_w = int(ex + float(ew / 2))
            iris_h = int(ey + float(eh / 2))
            irises.append([np.float32(iris_w), np.float32(iris_h)])

        return np.array(irises)
    
    def get_face_bbox(self, frame_gray):
        faces = self.face_cascade.detectMultiScale(frame_gray, 1.1, 4)
        
        if (len(faces) != 0):      
            if (len(self.prev_face) != 0):
                x_prev = self.prev_face[0]
                y_prev = self.prev_face[1]
                dists = np.zeros((len(faces),1))
                for k in range(len(faces)):
                    x = faces[k][0]
                    y = faces[k][1]
                    dists[k] = (x_prev-x)^2 + (y_prev-y)^2
                arg_min = np.argmin(dists)
                return faces[arg_min] # discard other people's faces
            else:
                return faces[0]
        else:
            return []
    
    def init_smoother(self, init_face):
        self.smooth_init = True
        self.sp = init_face
        self.sp2 = init_face
        self.sl = len(init_face)
        
    def double_exp_smoother(self, face, alpha = 0.35):
        if(self.smooth_init):
            # update
            self.sp = alpha * face + (1-alpha)*self.sp
            self.sp2 = alpha * self.sp + (1-alpha)*self.sp2
            new_face = np.rint(2*self.sp - self.sp2)
            return new_face.astype(int)
        else:
            return face


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
        
        self.fov = None
        self.distance_from_camera_to_screen = None
    
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

    def show_tracked_head(self, img, face, x_cm, y_cm, z_cm):
        if (len(face) >= 4):
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
            text_x = 'x: {}'.format(x_cm)
            text_y = 'y: {}'.format(y_cm)
            text_z = 'z: {}'.format(z_cm)
            cv2.putText(img, text_x, (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
            cv2.putText(img, text_y, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
            cv2.putText(img, text_z, (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
        cv2.imshow('img', img)
                
 
    def headposition(self, face, img, edge_correction):
        # some assumptions that are used when calculating distances and estimating horizontal fov
		# head width = 16 cm
		# head height = 19 cm
		# when initialized, user is approximately 60 cm from camera
        head_width_cm = 16
        head_height_cm = 19
        
        # angle between side of face and diagonal across
        head_small_angle = np.arctan(head_width_cm/head_height_cm)
        head_diag_cm = np.sqrt((head_width_cm*head_width_cm)+(head_height_cm*head_height_cm)) # diagonal of face in real space
        
        camheight_cam, camwidth_cam = img.shape[:2]
        
        sin_hsa = np.sin(head_small_angle) #precalculated sine
        cos_hsa = np.cos(head_small_angle) #precalculated cosine
        tan_hsa = np.tan(head_small_angle) #precalculated tan
        
        # estimate horizontal field of view of camera
        init_width_cam = face[2]
        init_height_cam = face[3]
        head_diag_cam = np.sqrt((init_width_cam*init_width_cam)+(init_height_cam*init_height_cam))
        
        # we use the diagonal of the faceobject to estimate field of view of the camera
        # we use the diagonal since this is less sensitive to errors in width or height
        head_width_cam = sin_hsa * head_diag_cam
        camwidth_at_default_face_cm = (camwidth_cam/head_width_cam) * head_width_cm
        
        if self.fov == None:
            # we assume user is sitting around 60 cm from camera (normal distance on a laptop)
            distance_to_screen = 45
        
            #calculate estimate of field of view
            fov_width = np.arctan((camwidth_at_default_face_cm/2)/distance_to_screen) * 2
        else:
            fov_width = self.fov * np.pi/180
            
        self.fov = fov_width * 180/np.pi
            
        # precalculate ratio between camwidth and distance
        tan_fov_width = 2 * np.tan(fov_width/2)
        
        # calculate cm-distance from screen
        z = (head_diag_cm*camwidth_cam)/(tan_fov_width*head_diag_cam)
		# to transform to z_3ds : z_3ds = (head_diag_3ds/head_diag_cm)*z
		# i.e. just use ratio
		
        w = face[2]
        h = face[3]
        fx = face[0] + w/2
        fy = face[1] + h/2
        
        if(edge_correction):
            # recalculate head_diag_cam, fx, fy
            margin = 11
            
            leftDistance = fx - w/2
            rightDistance = camwidth_cam - (fx + w/2)
            topDistance = fy - h/2
            bottomDistance = camheight_cam - (fy + h/2)
            
            onVerticalEdge = leftDistance < margin or rightDistance < margin
            onHorizontalEdge = topDistance < margin or bottomDistance < margin
            
            if(onHorizontalEdge):
                if(onVerticalEdge):
                    # we are in a corner, use previous diagonal as estimate, i.e. don't change head_diag_cam
                    onLeftEdge = leftDistance < margin
                    onTopEdge = topDistance < margin
                    
                    if (onLeftEdge):
                        fx = w - (head_diag_cam * sin_hsa/2)
                    else:
                        fx = fx - w/2 + (head_diag_cam * sin_hsa/2)
                    
                    if (onTopEdge):
                        fy = h - (head_diag_cam * cos_hsa/2)
                    else:
                        fy = fy - h/2 + (head_diag_cam * cos_hsa/2)
                else:
                    # we are on top or bottom edge of camera, use width insted of diagonal and correct y-position
                    # fix fy
                    if (topDistance < margin):
                        originalWeight = topDistance/margin
                        estimateWeight = (margin - topDistance)/margin
                        fy = h - (originalWeight*(h/2) + estimateWeight*((w/tan_hsa)/2))
                        head_diag_cam = estimateWeight*(2/sin_hsa) + originalWeight*(np.sqrt(w*w + h*h))
                    else:
                        originalWeight = bottomDistance/margin
                        estimateWeight = (margin - bottomDistance)/margin
                        fy = fy - h/2 + (originalWeight*(h/2) + estimateWeight*((w/tan_hsa)/2))
                        head_diag_cam = estimateWeight*(w/sin_hsa) + originalWeight*(np.sqrt(w*w + h*h))
            elif (onVerticalEdge):
                # we are on the sides of the camera, use height and correct x-position
                if (leftDistance < margin):
                    originalWeight = leftDistance/margin
                    estimateWeight = (margin - leftDistance)/margin
                    head_diag_cam = estimateWeight*(h/cos_hsa) + originalWeight*(np.sqrt(w*w + h*h))
                    fx = w - (originalWeight*(w/2) + estimateWeight*(h*tan_hsa/2))
                else:
                    originalWeight = rightDistance/margin
                    estimateWeight = (margin - rightDistance)/margin
                    head_diag_cam = estimateWeight*(h/cos_hsa) + originalWeight*(np.sqrt(w*w + h*h))
                    fx = fx - w/2 + (originalWeight*(w/2) + estimateWeight*(h*tan_hsa/2))
            else:
                head_diag_cam = np.sqrt(w*w + h*h)
        else:
            head_diag_cam = np.sqrt(w*w + h*h)
        
		# calculate cm-position relative to center of screen
        x = -((fx/camwidth_cam) - 0.5) * z * tan_fov_width
        y = -((fy/camheight_cam) - 0.5) * z * tan_fov_width * (camheight_cam/camwidth_cam)
		
		
		# Transformation from position relative to camera, to position relative to center of screen
        if self.distance_from_camera_to_screen == None:
            # default is 11.5 cm approximately
            y + 11.5
            
        else:
            y = y + self.distance_from_camera_to_screen
		
        
        return x, y, z
    
    def face_tracker(self):
        # Read the input image
        #img = cv2.imread('test.png')
        
        k = cv2.waitKey(30) & 0xff
        while k != 27:
            img = self.webcam.get_current_frame()        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            face = self.classifier.get_face_bbox(gray)
            face_x_cm, face_y_cm, face_z_cm = 0, 0, 0

            if (len(self.classifier.prev_face) != 0 and len(face) != 0):
                
                if (not self.classifier.smooth_init):
                    self.classifier.init_smoother(face)
                else:
                    face = self.classifier.double_exp_smoother(face,0.2)
                
                face_x_cm, face_y_cm, face_z_cm = self.headposition(face, img, True)
                message = {"eyeX": face_x_cm,
                           "eyeY": face_y_cm,
                           "eyeZ": face_z_cm}
                loop = asyncio.get_event_loop()
                loop.run_until_complete(tcp_echo_client(message, loop))

            
            # Display the output
            self.show_tracked_head(img,face,face_x_cm, face_y_cm, face_z_cm)
            k = cv2.waitKey(30) & 0xff
            self.classifier.prev_face = face
        self.webcam.release()
        cv2.destroyAllWindows()
            
    def main(self):
        self.face_tracker()
        print('finished lego main')
 
# run instance of Lego Tracker 
legoTracker = LegoTracker(image_source=0, classifier=CascadeClassifier(glasses = False),
                                 tracker=LucasKanadeTracker())
legoTracker.main()