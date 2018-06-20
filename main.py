import dlib
#from skimage import io
#from skimage.io import ImageCollection
#from skimage.io.video import CvVideo
import skimage.transform as tr
import numpy as np
import cv2
import os.path
from scipy.spatial import distance

import threading

##
memory = {}
##

###
def _rect2bb(rect):
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y
    return(x,y,w,h)

def _shape2np(shape, dtype="int"):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return(coords)

def _align_landmark(image, predictor, face_rect):
    dim = 500
    border = 100
    mask0 = [39, 42, 57]
    rect = face_rect
    mask = np.array(mask0)

    landmarks = np.array(list(map(lambda p: [p.x, p.y], predictor(image, rect).parts())))
    proper_landmarks = border + dim * np.load(os.path.join("/home/arhin/FLD/", 'face_template.npy'))[mask]
    A = np.hstack([landmarks[mask], np.ones((3, 1))]).astype(np.float64)
    B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
    T = np.linalg.solve(A, B).T
    wrapped = tr.warp(  image,
                        tr.AffineTransform(T).inverse,
                        output_shape=(dim + 2 * border, dim + 2 * border),
                        order=0,
                        mode='constant',
                        cval=0,
                        clip=True,
                        preserve_range=True)
    return wrapped/255.0
###

def save_human():
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()

    video_capture = cv2.VideoCapture(1)
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    face_descriptor = ()
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dets = detector(gray, 1)

        for k, d in enumerate(dets):
            shape = sp(frame,d)

            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            #print(face_descriptor)

            shape = _shape2np(shape)
            (x,y,w,h) = _rect2bb(d)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

            for (x,y) in shape:
                cv2.circle(frame, (x,y), 2, (255,0,0), 3)
            


        aligned = _align_landmark(frame, sp, dets)


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            a = input()
            if(a==""): 
                break
            memory[face_descriptor] = str(a)
            break


    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()




def stream_detect():
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()

    video_capture = cv2.VideoCapture(0)
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    while True:
        # Capture frame-by-frame
        
        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray frame

        dets = detector(gray, 1) # detect faces in the grayscale frame (rects)
        
        
        for k, d in enumerate(dets):
            #crop = gray[d.top():d.bottom(), d.left():d.right()] # cropped face from gray frame
            aligned = (255 * _align_landmark(gray, sp, d)).astype(np.uint8)
            _d = detector(aligned, 1)
            
            for _, __d in enumerate(_d):
                shape = sp(aligned, __d)
                ### draw face landmarks
                np_shape = _shape2np(shape) # NumPy array of landmarks
                for (x,y) in np_shape:
                    cv2.circle(aligned, (x,y), 2, (0,0,0), 3)
                #/# draw face landmarks

                ### find a similar face in the db
                '''face_descriptor = facerec.compute_face_descriptor(frame, shape)
                cust_list = memory.keys()
                for i in cust_list:
                    dst = distance.euclidean(face_descriptor, i)
                    if(dst<0.6):
                        print(dst)
                        print(memory[i])
                        break
                print(face_descriptor)'''
                #/# find a similar face in the db

                cv2.imshow('Aligned'+str(_), aligned)

            #shape = sp(gray, d)
            



            ### draw face rectangle
            (x,y,w,h) = _rect2bb(d)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
            #/# draw face rectangle

            '''### draw face landmarks
            np_shape = _shape2np(shape) # NumPy array of landmarks
            for (x,y) in np_shape:
                cv2.circle(frame, (x,y), 2, (255,0,0), 3)
            #/# draw face landmarks'''


        


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print("1. Save human")
    print("2. Stream detection")
    print("0. Exit")
    while True:
        print(">> ",end='')
        a = input()
        if(int(a) == 1):
            save_human()
        if(int(a) == 2):
            stream_detect()
        if(int(a) == 0):
            break