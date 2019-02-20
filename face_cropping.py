import os
import glob
import cv2
import utils
import pickle
import numpy as np
import client_thrift
from sklearn.utils import shuffle
from sklearn import model_selection, svm
from scipy import spatial
from sklearn.linear_model import LogisticRegression, SGDClassifier
from face_detection import detect_face


data_path = 'data/mydata'
lists = ['colin', 'silva', 'blix', 'donalt']


def dump_data():
    for person in lists:
        person_images = os.path.join(data_path, person)
        for image_path in glob.glob(os.path.join(person_images, '*.jpg')):
            image = utils.load_rgb_image(image_path)
            ret, _ = detect_face(image)
            maxx = 0
            face = None
            for bb in ret:
                l, t, r, b = bb
                if (b - t) * (r - l) > maxx:
                    face = image[t:b,l:r]
            if face is None:
                os.remove(image_path)
                continue
            cv2.imwrite(image_path, face[:,:,::-1])

dump_data()
            
