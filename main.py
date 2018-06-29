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

###

k_euclid = 0.6

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
        np.load(os.path.join(os.curdir, 'face_template.npy'))[mask]
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
    names = [""]*100
    dst = [""]*100
    face_descs = [""]*100

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
                        self.frame, "("+str(k)+")"+self.names[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
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
                if key == ord('x'):
                    self.running = False

        video_capture.release()
        cv2.destroyAllWindows()

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
                            face_descriptor = facerec.compute_face_descriptor(
                                aligned, shape)
                            self.face_descs[k] = face_descriptor
                            cust_list = memory.keys()
                            for i in cust_list:
                                dst = distance.euclidean(face_descriptor, i)
                                if(dst < k_euclid):
                                    # print(dst)
                                    self.names[k] = memory[i]
                                    self.dst[k] = "("+str(round(dst, 2))+")"
                                    break
                            # /# find a similar face in the db

            else:
                time.sleep(0.2)


class StreamDetector():

    frame = None
    cam_id = None
    running = False

    dets = None
    names = [""]*100
    dst = [""]*100

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
                        self.frame, "("+str(k)+")"+self.names[k] + " " + self.dst[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # /# draw face rectangle and name

            cv2.imshow('Video', self.frame)
            _, self.frame = video_capture.read()
            key = cv2.waitKey(1)
            if key:
                if key == ord('x'):
                    self.running = False

        video_capture.release()
        cv2.destroyAllWindows()

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
                    self.names[k] = 'Face'
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
                                if(dst < k_euclid):
                                    # print(dst)
                                    self.names[k] = memory[i]
                                    self.dst[k] = "("+str(round(dst, 2))+")"
                                    break
                            # /# find a similar face in the db

            else:
                time.sleep(0.2)


class PhotoSaver():

    running = False
    face_descs = [""]*100

    def start(self):
        self.main()

    def _sort_dirlist(self, dirlist):
        sorted_dirlist = []
        for file in dirlist:
            if (
                file.endswith(".jpg") or
                file.endswith(".png")
            ):
                sorted_dirlist.append(file)
        return sorted_dirlist

    def main(self):
        dirlist = os.listdir(os.curdir + "/photos")
        dirlist = self._sort_dirlist(dirlist)

        iter = 0
        self.running = True

        while self.running:
            path = os.path.join(os.curdir + "/photos", dirlist[iter])
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()
            facerec = dlib.face_recognition_model_v1(
                "dlib_face_recognition_resnet_model_v1.dat")

            names = [""]*100
            dsts = [""]*100

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray frame

            # detect faces in the grayscale frame (rects)
            self.dets = detector(gray, 1)
            if(self.dets):
                for k, d in enumerate(self.dets):
                    aligned = (255 * _align_landmark(img, sp, d)
                               ).astype(np.uint8)
                    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    _d = detector(aligned, 1)
                    names[k] = 'Face'
                    dsts[k] = "(None)"
                    if _d:
                        _, __d = list(enumerate(_d))[0]

                        shape = sp(gray_aligned, __d)

                        if True:
                            # find a similar face in the db
                            face_descriptor = facerec.compute_face_descriptor(
                                aligned, shape)
                            self.face_descs[k] = face_descriptor
                            cust_list = memory.keys()
                            for i in cust_list:
                                dst = distance.euclidean(face_descriptor, i)
                                if(dst < k_euclid):
                                    # print(dst)
                                    names[k] = memory[i]
                                    dsts[k] = "("+str(round(dst, 2))+")"
                                    break
                            # /# find a similar face in the db
                            # draw face rectangle and name
                            (x, y, w, h) = _rect2bb(d)
                            cv2.rectangle(img, (x, y),
                                          (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(
                                img, "("+str(k)+")"+names[k] + " " + dsts[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            # /# draw face rectangle and name

            cv2.imshow('Photos', img)
            key = cv2.waitKey(0)
            if key:
                if key == ord('q'):
                    if(iter > 0):
                        iter -= 1
                    else:
                        iter = (len(dirlist)-1)
                elif key == ord('e'):
                    if(iter < (len(dirlist)-1)):
                        iter += 1
                    else:
                        iter = 0

                elif key == ord('s'):
                    while True:
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

                elif key == ord('x'):
                    self.running = False

        cv2.destroyAllWindows()


class PhotoDetector():

    running = False

    def start(self):
        self.main()

    def _sort_dirlist(self, dirlist):
        sorted_dirlist = []
        for file in dirlist:
            if (
                file.endswith(".jpg") or
                file.endswith(".png")
            ):
                sorted_dirlist.append(file)
        return sorted_dirlist

    def main(self):
        dirlist = os.listdir(os.curdir + "/photos")
        dirlist = self._sort_dirlist(dirlist)

        iter = 0
        self.running = True

        while self.running:
            path = os.path.join(os.curdir + "/photos", dirlist[iter])
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            detector = dlib.get_frontal_face_detector()
            facerec = dlib.face_recognition_model_v1(
                "dlib_face_recognition_resnet_model_v1.dat")

            names = [""]*100
            dsts = [""]*100

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # gray frame

            # detect faces in the grayscale frame (rects)
            self.dets = detector(gray, 1)
            if(self.dets):
                for k, d in enumerate(self.dets):
                    aligned = (255 * _align_landmark(img, sp, d)
                               ).astype(np.uint8)
                    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    _d = detector(aligned, 1)
                    names[k] = 'Face'
                    dsts[k] = "(None)"
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
                                if(dst < k_euclid):
                                    # print(dst)
                                    names[k] = memory[i]
                                    dsts[k] = "("+str(round(dst, 2))+")"
                                    break
                            # /# find a similar face in the db
                            # draw face rectangle and name
                            (x, y, w, h) = _rect2bb(d)
                            cv2.rectangle(img, (x, y),
                                          (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(
                                img, "("+str(k)+")"+names[k] + " " + dsts[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            # /# draw face rectangle and name

            cv2.imshow('Photos', img)
            key = cv2.waitKey(0)
            if key:
                if key == ord('q'):
                    if(iter > 0):
                        iter -= 1
                    else:
                        iter = (len(dirlist)-1)
                if key == ord('e'):
                    if(iter < (len(dirlist)-1)):
                        iter += 1
                    else:
                        iter = 0
                if key == ord('x'):
                    self.running = False

        cv2.destroyAllWindows()


def _save_fdb():
    global memory
    file = open("fdb.b", 'wb')
    cPickle.dump(memory, file)
    file.close()


def _load_fdb():
    global memory
    try:
        file = open("fdb.b", 'rb')
        memory = cPickle.load(file)
        file.close()
    except:
        print("Loading error")


def main():
    cam_id = 0
    _load_fdb()
    state = "0"

    StreamS = None
    StreamD = None
    PhotoS = None
    PhotoD = None

    while True:
        if state == "exit":
            _save_fdb()
            break
        elif state == "0":
            print("1. Stream")
            print("2. Photos")
            print("0. Exit")
            print(">> ", end='')
            i = input()
            if i == "1":
                state = "1"
            elif i == "2":
                state = "2"
            elif i == "0":
                state = "exit"
            else:
                print("Missing command")

        elif state == "1":
            StreamS = StreamSaver(cam_id)
            StreamD = StreamDetector(cam_id)
            print("1. (Stream) Save face")
            print("2. (Stream) Face detector")
            print("0. Back")
            print(">> ", end='')
            i = input()
            if i == "1":
                state = "11"
            elif i == "2":
                state = "12"
            elif i == "0":
                state = "0"
            else:
                print("Missing command")
        elif state == "2":
            PhotoS = PhotoSaver()
            PhotoD = PhotoDetector()
            print("1. (Photos) Save face")
            print("2. (Photos) Face detector")
            print("0. Back")
            print(">> ", end='')
            i = input()
            if i == "1":
                state = "21"
            elif i == "2":
                state = "22"
            elif i == "0":
                state = "0"
            else:
                print("Missing command")

        elif state == "11":
            StreamS.start()
            state = "1"
        elif state == "12":
            StreamD.start()
            state = "1"
        elif state == "21":
            PhotoS.start()
            state = "2"
        elif state == "22":
            PhotoD.start()
            state = "2"
        else:
            print("Missing state")
            state = "0"


if __name__ == '__main__':
    main()
