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


class S_D_Base():
    running = False

    frame = None
    dets = None
    names = [""]*100
    dsts = [""]*100
    face_descs = [""]*100

    def init_detector(self):
        sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        facerec = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat")
        return (sp, detector, facerec)

    def find_similar_in_fdb(self, face_descriptor, k):
        # find a similar face in the db
        cust_list = memory.keys()
        for i in cust_list:
            dst = distance.euclidean(face_descriptor, i)
            if(dst < k_euclid):
                # print(dst)
                self.names[k] = memory[i]
                self.dsts[k] = "("+str(round(dst, 2))+")"
                break
        # /# find a similar face in the db

    def draw_face_rect(self, rect, k):
        # draw face rectangle and name
        (x, y, w, h) = self._rect2bb(rect)
        cv2.rectangle(self.frame, (x, y),
                      (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            self.frame, "("+str(k)+")"+self.names[k] + " " + self.dsts[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # /# draw face rectangle and name

    def _rect2bb(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right()-x
        h = rect.bottom()-y
        return(x, y, w, h)

    def _shape2np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return(coords)

    def _align_landmark(self, image, predictor, face_rect):
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


class StreamDetector(S_D_Base):

    cam_id = None

    def __init__(self, cam_id):
        self.cam_id = cam_id

    def start(self):
        sec_thread = threading.Thread(target=self.secondary_thread)
        sec_thread.start()
        self.main_thread()
        sec_thread.join()

    def main_thread(self):
        video_capture = cv2.VideoCapture(self.cam_id)
        _, self.frame = video_capture.read()

        self.running = True
        while self.running:
            if self.dets:
                for k, d in enumerate(self.dets):
                    self.draw_face_rect(d, k)

            cv2.imshow('Video', self.frame)
            _, self.frame = video_capture.read()

            key = cv2.waitKey(1)
            if key:
                if key == ord('s'):
                    while True:
                        print("Face number:", end='')
                        i = input()
                        print("Owner's name:", end='')
                        name = input()
                        if(name == "" or i == ""):
                            print("Error adding")
                            self.running = False
                            break
                        if(self.face_descs[int(i)]):
                            memory[self.face_descs[int(i)]] = str(name)
                        else:
                            print("Error adding")
                            self.running = False
                            break

                elif key == ord('x'):
                    self.running = False

        video_capture.release()
        cv2.destroyAllWindows()

    def secondary_thread(self):
        sp, detector, facerec = self.init_detector()

        self.running = True
        while self.running:
            if self.frame is None:
                continue
            time.sleep(0.1)

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # gray frame

            # detect faces in the grayscale frame (rects)
            self.dets = detector(gray, 1)
            if self.dets:

                for k, d in enumerate(self.dets):
                    aligned = (255 * self._align_landmark(self.frame, sp, d)
                               ).astype(np.uint8)
                    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    _d = detector(aligned, 1)
                    self.names[k] = 'Face '+str(k)
                    self.dsts[k] = "(None)"
                    if _d:
                        _, __d = list(enumerate(_d))[0]

                        shape = sp(gray_aligned, __d)

                        self.face_descs[k] = facerec.compute_face_descriptor(
                            aligned, shape)
                        self.find_similar_in_fdb(self.face_descs[k], k)

            else:
                time.sleep(0.2)


class PhotoDetector(S_D_Base):

    def start(self):
        self.main_thread()

    def _sort_dirlist(self, dirlist):
        sorted_dirlist = []
        for file in dirlist:
            if (
                file.endswith(".jpg") or
                file.endswith(".png")
            ):
                sorted_dirlist.append(file)
        return sorted_dirlist

    def main_thread(self):
        dirlist = os.listdir(os.curdir + "/photos")
        dirlist = self._sort_dirlist(dirlist)

        iter = 0

        self.running = True
        while self.running:
            path = os.path.join(os.curdir + "/photos", dirlist[iter])
            self.frame = cv2.imread(path, cv2.IMREAD_COLOR)

            sp, detector, facerec = self.init_detector()

            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)  # gray frame

            # detect faces in the grayscale frame (rects)
            self.dets = detector(gray, 1)
            if(self.dets):
                for k, d in enumerate(self.dets):
                    aligned = (255 * self._align_landmark(self.frame, sp, d)
                               ).astype(np.uint8)
                    gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    _d = detector(aligned, 1)
                    self.names[k] = 'Face'
                    self.dsts[k] = "(None)"
                    if _d:
                        _, __d = list(enumerate(_d))[0]

                        shape = sp(gray_aligned, __d)

                        self.face_descs[k] = facerec.compute_face_descriptor(
                            aligned, shape)
                        self.find_similar_in_fdb(self.face_descs[k], k)
                        self.draw_face_rect(d, k)

            cv2.imshow('Photos', self.frame)
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
                        if(name == "" or i == ""):
                            print("Error adding")
                            self.running = False
                            break
                        if(self.face_descs[int(i)]):
                            memory[self.face_descs[int(i)]] = str(name)
                        else:
                            print("Error adding")
                            self.running = False
                            break

                elif key == ord('x'):
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

    StreamD = None
    PhotoD = None

    while True:
        if state == "exit":
            _save_fdb()
            break

        elif state == "0":
            StreamD = StreamDetector(cam_id)
            PhotoD = PhotoDetector()
            print("1. Stream Detector")
            print("2. Photo Detector")
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
            StreamD.start()
            state = "0"
        elif state == "2":
            PhotoD.start()
            state = "0"

        else:
            print("Missing state")
            state = "0"


if __name__ == '__main__':
    main()
