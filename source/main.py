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
import csv
from scipy import spatial
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
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return abs(similarity) > 0.601


def evaluate(predictions, labels):
    TP, FP, FN = 0, 0, 0
    for i in range(len(predictions)):
        if int(predictions[i]) == 1 and labels[i] == 1:
            TP += 1
        elif int(predictions[i]) == 1 and labels[i] == 0:
            FP += 1
        elif int(predictions[i]) == 0 and labels[i] == 1:
            FN += 1
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * precision * recall / (recall + precision)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1: ", F1_score)
    except:
        pass


target_face = get_target_face()
target_embedding = np.array(client_thrift.get_emb_numpy([target_face])[0])

filenames, labels = get_annotation_data()

predictions = []
for idx, fname in enumerate(filenames):
    image_path = os.path.join(config.PUBLIC_TEST_PATH, fname + '.jpg')
    image = utils.load_rgb_image(image_path)
    pred, bb = 0, None
    if image is not None:
        bbs, _ = detect_face(image.copy())
        if len(bbs) == 0: 
            print("No face found in: ", image_path)
        for idx2, bounding_box in enumerate(bbs):
            l, t, r, b = bounding_box
            face = image[t:b,l:r]
            face_emb = utils.get_feature_vec(face, fname + str(idx2))
            if is_matched(target_embedding, face_emb):
                pred = 1
                bb = bounding_box
    if pred == 0:
        predictions.append([fname, 0, 0, 0, 0, pred])
    else:
        tmp = [fname]
        tmp.extend(bb)
        tmp.append(pred)
        predictions.append(tmp)
    if idx % 10 == 0 and idx != 0:
        print("Image " + str(idx) + "th")
    if idx > 1100:
        break

evaluate(np.array(predictions)[:,-1], labels)

csv_content = [["image", "x1", "y1", "x2", "y2", "result"]]
csv_content.extend(predictions)
with open('output.csv', 'w') as fn:
    csv.writer(fn).writerows(csv_content)
