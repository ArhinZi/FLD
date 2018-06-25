import dlib
#from skimage import io
#from skimage.io import ImageCollection
#from skimage.io.video import CvVideo
import skimage.transform as tr
import numpy as np
import cv2
import os.path
from scipy.spatial import distance
from PIL import Image

import threading

##
cam_id = 1
memory = {}
##

###


def _rect2bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right()-x
    h = rect.bottom()-y
    return(x, y, w, h)


def _shape2np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return(coords)


def _align_landmark(image, predictor, face_rect):
    dim = 500
    border = 100
    mask0 = [39, 42, 57]
    rect = face_rect
    mask = np.array(mask0)

    landmarks = np.array(
        list(map(lambda p: [p.x, p.y], predictor(image, rect).parts())))
    proper_landmarks = border + dim * \
        np.load(os.path.join("/home/arhin/FLD/", 'face_template.npy'))[mask]
    A = np.hstack([landmarks[mask], np.ones((3, 1))]).astype(np.float64)
    B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
    T = np.linalg.solve(A, B).T
    wrapped = tr.warp(image,
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

    video_capture = cv2.VideoCapture(cam_id)
    facerec = dlib.face_recognition_model_v1(
        "dlib_face_recognition_resnet_model_v1.dat")
    while True:
        # Capture frame-by-frame

        _, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # gray frame

        dets = detector(gray, 1)  # detect faces in the grayscale frame (rects)
        if(len(dets) == 0):
            for i in range(10):
                try:
                    cv2.destroyWindow("Aligned"+str(i))
                except:
                    pass
        for _, d in enumerate(dets):
            aligned = (255 * _align_landmark(frame, sp, d)).astype(np.uint8)
            gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            _d = detector(aligned, 1)
            try:
                k, __d = list(enumerate(_d))[0]

                shape = sp(gray_aligned, __d)

                name = 'Aligned'+str(k)
                try:
                    face_descriptor = facerec.compute_face_descriptor(
                        aligned, shape)  # find face descriptor
                except:
                    pass

                # draw face landmarks
                np_shape = _shape2np(shape)  # NumPy array of landmarks
                for (x, y) in np_shape:
                    cv2.circle(aligned, (x, y), 2, (0, 0, 0), 3)
                # /# draw face landmarks

                cv2.imshow(name, aligned)
            except:
                pass

            # draw face rectangle
            (x, y, w, h) = _rect2bb(d)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # /# draw face rectangle

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key:
            if key == ord('q'):
                break
            if key == ord('s'):
                print("Face number:", end='')
                i = input()
                print("Owner's name:", end='')
                name = input()
                if(name == ""):
                    break
                try:
                    memory[face_descriptor] = str(name)
                except:
                    print("Error!!!")
                break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


class StreamDetector():
    frame = None
    cam_id = None
    running = False

    dets = None
    names = [""]*20

    def __init__(self, cam_id):
        self.cam_id = cam_id

    def _detector(self):
        lock = threading.Lock()
        sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        facerec = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat")

        while self.running:
            if self.frame is None:
                continue
            
            
            lock.acquire()

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # gray frame


            self.dets = detector(gray, 1)  # detect faces in the grayscale frame (rects)
            for k, d in enumerate(self.dets):
                aligned = (255 * _align_landmark(self.frame, sp, d)
                        ).astype(np.uint8)
                gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                _d = detector(aligned, 1)
                self.names[k] = 'Face '+str(k)
                try:
                    _, __d = list(enumerate(_d))[0]

                    shape = sp(gray_aligned, __d)
                    
                    try:
                        # find a similar face in the db
                        face_descriptor = facerec.compute_face_descriptor(
                            aligned, shape)
                        cust_list = memory.keys()
                        for i in cust_list:
                            dst = distance.euclidean(face_descriptor, i)
                            if(dst < 0.5):
                                # print(dst)
                                self.names[k] = memory[i]
                                break
                        # /# find a similar face in the db
                    except:
                        pass

                    # # draw face landmarks
                    # np_shape = _shape2np(shape)  # NumPy array of landmarks
                    # for (x, y) in np_shape:
                    #     cv2.circle(aligned, (x, y), 2, (0, 0, 0), 3)
                    # # /# draw face landmarks

                    #cv2.imshow(name, aligned)
                except:
                    pass

                

            lock.release()

    def start(self):
        lock = threading.Lock()
        video_capture = cv2.VideoCapture(self.cam_id)
        self.running = True
        _, self.frame = video_capture.read()
        while True:
            lock.acquire()
            try:
                for k, d in enumerate(self.dets):
                    # draw face rectangle
                    (x, y, w, h) = _rect2bb(d)
                    cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(self.frame, self.names[k], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    # /# draw face rectangle
            except:
                pass
            try:
                cv2.imshow('Video', self.frame)
            except:
                pass
            _, self.frame = video_capture.read()
            key = cv2.waitKey(1)
            if key:
                if key == ord('q'):
                    break
            lock.release()
        video_capture.release()
        cv2.destroyAllWindows()
        self.running = False


if __name__ == '__main__':
    print("1. Save human")
    print("2. Stream detection")
    print("0. Exit")
    streamer = StreamDetector(cam_id)
    while True:
        try:
            print(">> ", end='')
            a = input()
            if(int(a) == 1):
                save_human()
            if(int(a) == 2):
                main = threading.Thread(target=streamer.start)
                det = threading.Thread(target=streamer._detector)
                main.start()
                det.start()

                main.join()
            if(int(a) == 0):
                break
        except:
            print("Error!!!")
