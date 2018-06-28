import dlib
# from skimage import io
# from skimage.io import ImageCollection
# from skimage.io.video import CvVideo
import skimage.transform as tr
import numpy as np
import cv2
import os.path
from scipy.spatial import distance
from PIL import Image
import time
import _pickle as cPickle

import threading


memory = {}
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


class StreamSaver():

    frame = None
    cam_id = None
    running = False

    dets = None
    names = [""]*20
    face_descs = [""]*20

    def __init__(self, cam_id):
        self.cam_id = cam_id

    def start(self):
        det = threading.Thread(target=self.s_detector_thread)
        det.start()
        self.main_thread()
        det.join()

    def main_thread(self):
        video_capture = cv2.VideoCapture(self.cam_id)
        self.running = True
        _, self.frame = video_capture.read()
        while self.running:

            if self.dets:
                for k, d in enumerate(self.dets):
                    # draw face rectangle and name
                    (x, y, w, h) = _rect2bb(d)
                    cv2.rectangle(self.frame, (x, y),
                                  (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        self.frame, self.names[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # /# draw face rectangle and name

            cv2.imshow('Video', self.frame)
            _, self.frame = video_capture.read()
            key = cv2.waitKey(1)
            if key:
                if key == ord('s'):
                    print("Face number:", end='')
                    i = input()
                    print("Owner's name:", end='')
                    name = input()
                    if(name == ""):
                        self.running = False
                    if(self.face_descs[int(i)]):
                        memory[self.face_descs[int(i)]] = str(name)
                    else:
                        print("Error adding")
                    self.running = False
                    break

            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break

        video_capture.release()
        cv2.destroyAllWindows()
        self.running = False

    def s_detector_thread(self):
        sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        facerec = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat")

        while self.running:
            if self.frame is None:
                continue

            time.sleep(0.1)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # gray frame

            # detect faces in the grayscale frame (rects)
            self.dets = detector(gray, 1)
            if(self.dets):
                for k, d in enumerate(self.dets):
                    aligned = (255 * _align_landmark(self.frame, sp, d)
                               ).astype(np.uint8)
                    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    _d = detector(aligned, 1)
                    self.names[k] = 'Face '+str(k)
                    if _d:
                        _, __d = list(enumerate(_d))[0]

                        shape = sp(gray_aligned, __d)

                        if True:
                            # find a similar face in the db
                            self.face_descs[k] = facerec.compute_face_descriptor(
                                aligned, shape)
                            # /# find a similar face in the db

            else:
                time.sleep(0.2)


class StreamDetector():

    frame = None
    cam_id = None
    running = False

    dets = None
    names = [""]*20
    dst = [""]*20

    def __init__(self, cam_id):
        self.cam_id = cam_id

    def start(self):
        det = threading.Thread(target=self.detector_thread)
        det.start()
        self.main_thread()
        det.join()

    def main_thread(self):
        video_capture = cv2.VideoCapture(self.cam_id)
        self.running = True
        _, self.frame = video_capture.read()
        while self.running:

            if self.dets:
                for k, d in enumerate(self.dets):
                    # draw face rectangle and name
                    (x, y, w, h) = _rect2bb(d)
                    cv2.rectangle(self.frame, (x, y),
                                  (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        self.frame, self.names[k] + " " + self.dst[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # /# draw face rectangle and name

            cv2.imshow('Video', self.frame)
            _, self.frame = video_capture.read()
            cv2.waitKey(1)

            if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break

        video_capture.release()
        cv2.destroyAllWindows()
        self.running = False

    def detector_thread(self):
        sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        facerec = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat")

        while self.running:
            if self.frame is None:
                continue
            time.sleep(0.1)
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # gray frame

            # detect faces in the grayscale frame (rects)
            self.dets = detector(gray, 1)
            if(self.dets):
                for k, d in enumerate(self.dets):
                    aligned = (255 * _align_landmark(self.frame, sp, d)
                               ).astype(np.uint8)
                    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    _d = detector(aligned, 1)
                    self.names[k] = 'Face '+str(k)
                    self.dst[k] = "(None)"
                    if _d:
                        _, __d = list(enumerate(_d))[0]

                        shape = sp(gray_aligned, __d)

                        if True:
                            # find a similar face in the db
                            face_descriptor = facerec.compute_face_descriptor(
                                aligned, shape)
                            cust_list = memory.keys()
                            for i in cust_list:
                                dst = distance.euclidean(face_descriptor, i)
                                if(dst < 0.5):
                                    # print(dst)
                                    self.names[k] = memory[i]
                                    self.dst[k] = "("+str(round(dst, 2))+")"
                                    break
                            # /# find a similar face in the db

            else:
                time.sleep(0.2)


def _save_db():
    global memory
    file = open("fdb.b", 'wb')
    cPickle.dump(memory, file)
    file.close()


def _load_db():
    global memory
    try:
        file = open("fdb.b", 'rb')
        memory = cPickle.load(file)
        file.close()
    except:
        print("Loading error")


def main():
    cam_id = 0
    _load_db()
    print("1. Save human")
    print("2. Stream detection")
    print("0. Exit")

    while True:
        StreamS = StreamSaver(cam_id)
        StreamD = StreamDetector(cam_id)
        print(">> ", end='')
        a = input()
        try:
            if(int(a) == 1):
                StreamS.start()
            elif(int(a) == 2):
                StreamD.start()

            elif(int(a) == 0):
                _save_db()
                break
            else:
                print("Missing command")
        except ValueError:
            print("Invalid srting")


if __name__ == '__main__':
    main()
