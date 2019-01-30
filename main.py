import cv2
from imageio import imread
import os
import glob
import client_thrift
import utils
import config
from face_detection import detect_face
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def get_target_face():
    rgb_image = imread(config.SAMPLE_IMAGE_PATH)
    ret, _ = detect_face(rgb_image)
    if len(ret) <= 0:
        raise Exception('no face in sample image')
    l, t, r, b = ret[0]
    return rgb_image[t:b,l:r]


def get_annotation_data():
    data = pd.read_csv(config.ANNOTATION_PATH).values
    filenames = data[:,0]
    labels = data[:,-1]
    return filenames, labels


def is_matched(emb1, emb2):
    distance = np.sqrt(np.sum(np.square(emb1 - emb2)))
    if emb1.shape[0] == 2048:
        return distance < 110
    elif emb1.shape[0] == 128:
        return distance < 1.02
    else:
        return False


def evaluate(predictions, labels):
    TP, FP, FN = 0, 0, 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and labels[i] == 1:
            TP += 1
        elif predictions[i] == 1 and labels[i] == 0:
            FP += 1
        elif predictions[i] == 0 and labels[i] == 1:
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (recall + precision)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1_score)


target_face = get_target_face()
target_embedding = np.array(client_thrift.get_emb_numpy([target_face])[0])

filenames, labels = get_annotation_data()

predictions = []
for idx, fname in enumerate(filenames):
    image_path = os.path.join(config.PUBLIC_TEST_PATH, fname + '.jpg')
    image = cv2.imread(image_path)
    if image is None:
        print(image_path)
        continue
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bbs, _ = detect_face(image.copy())
    if len(bbs) == 0: predictions.append(0)
    pred = 0
    for bounding_box in bbs:
        l, t, r, b = bounding_box
        face = image[t:b,l:r]
        face_emb = np.array(client_thrift.get_emb_numpy([face])[0])
        if is_matched(target_embedding, face_emb):
            pred = 1

    predictions.append(pred)
    if idx % 100 == 0 and idx != 0:
        print("Image " + str(idx) + "th")
        evaluate(predictions[:idx], labels[:idx])
