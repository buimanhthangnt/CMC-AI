import numpy as np
import cv2
import pickle
import pandas as pd
import config
import os
from face_detection import detect_face
import client_thrift
import utils

def get_annotation_data():
    data = pd.read_csv(config.ANNOTATION_PATH).values
    filenames = data[:,0]
    labels = data[:,-1]
    return filenames, labels

filenames, labels = get_annotation_data()
features = []
for fname in filenames:
    image_path = os.path.join(config.PUBLIC_TEST_PATH, fname + '.jpg')
    print(image_path)
    image = utils.load_rgb_image(image_path)
    if image is None:
        features.append([])
        continue
    bbs, _ = detect_face(image.copy())
    if len(bbs) == 0:
        print("No face found in: ", image_path)
        features.append([])
    else:
        vecs = []
        for bounding_box in bbs:
            l, t, r, b = bounding_box
            face = image[t:b,l:r]
            face_emb = utils.get_feature_vec(face, fname)
            print(len(face_emb))
            vecs.append(face_emb)
        features.append(vecs)
print(len(features))
pickle.dump(features, open('data/publictest_facenetfeatures.p', 'wb'))

