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
import pickle as pkl
from scipy import spatial

def get_target_face():
    rgb_image = imread(config.SAMPLE_IMAGE_PATH)
    ret, _ = detect_face(rgb_image)
    if len(ret) <= 0:
        raise Exception('no face in sample image')
    l, t, r, b = ret[0]
    return rgb_image[t:b,l:r]


def get_annotation_data():
    data = pd.read_csv(config.ANNOTATION_PATH).values
    labels = data[:,-1]
    imgs = data[:,0]
    return labels, imgs


def is_matched_euclidean(emb1, emb2):
    distance = np.sqrt(np.sum(np.square(emb1 - emb2)))
    if emb1.shape[0] == 2048:
        return distance,distance < 114
    elif emb1.shape[0] == 128:
        return distance,distance < 0.78
    else:
        return False

def is_matched_cosine(emb1, emb2):
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return similarity, abs(similarity) > 0.601

def is_matched_combine(emb1, emb2):
    cosine = 1 - spatial.distance.cosine(emb1, emb2)
    distance = np.sqrt(np.sum(np.square(emb1 - emb2)))
    if emb1.shape[0] == 2048:
        return distance < 120 and abs(cosine) > 0.58
    elif emb1.shape[0] == 128:
        return distance < 0.78 and abs(cosine) > 0.58
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

labels, imgs = get_annotation_data()
if target_embedding.shape[0] ==2048: 
    test_features = pkl.load(open("data/publictest_resnetfeatures.p","rb"))
else: test_features = pkl.load(open("data/publictest_facenetfeatures.p","rb"))

predictions = []
for idx, img in enumerate(test_features):
    if len(img)==0:
        predictions.append(0)
    pred = 0    
    for vec in img:
        sim,matched = is_matched_cosine(target_embedding, vec)
        if matched:
            pred = 1
    #     print(idx,sim)
    # if pred==0 and labels[idx]==1 and len(img) != 0:
    #     try:
    #         img = cv2.imread(os.path.join(config.PUBLIC_TEST_PATH, imgs[idx] + '.jpg'))
    #         cv2.imshow('result', img )
    #         cv2.waitKey(0)
    #     except:
    #         pass       

    predictions.append(pred)
    if idx % 10 == 0 and idx != 0:
        print("Image " + str(idx) + "th")
        evaluate(predictions[:idx], labels[:idx])
