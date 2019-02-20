import cv2
from imageio import imread
import os
import glob
import client_thrift
import utils
import config
import pickle
from face_detection import detect_face
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pickle as pkl
from scipy import spatial
import imutils


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


def euclide_dist(emb1, emb2):
    distance = np.sqrt(np.sum(np.square(emb1 - emb2)))
    return distance


def cosine_dist(emb1, emb2):
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return similarity


def is_matched_cosine(emb1, emb2):
    similarity = cosine_dist(emb1, emb2)
    return similarity, abs(similarity) > 0.601


def is_matched_svm(emb_target, emb):
    euc = euclide_dist(emb_target, emb)
    cos = cosine_dist(emb_target, emb)
    gs = pickle.load(open('data/model/gs.pkl', 'rb'))
    pred = gs.predict_proba([[euc, cos]])[0]
    # print(gs.predict_proba([[euc, cos]])[0])
    return pred[1] >= 0.55


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
test_features = pkl.load(open("data/publictest_resnetfeatures.p","rb"))

predictions = []
for idx, img in enumerate(test_features):
    if len(img)==0:
        predictions.append(0)
    pred = 0    
    for vec in img:
        matched = is_matched_svm(target_embedding, vec)
        if matched:
            pred = 1
    # if pred==0 and labels[idx]==1 and len(img) != 0:
    #     try:
    #         img = cv2.imread(os.path.join(config.PUBLIC_TEST_PATH, imgs[idx] + '.jpg'))
    #         img = imutils.resize(img, height=600)
    #         cv2.imshow('result', img )
    #         cv2.waitKey(0)
    #     except:
    #         pass       

    predictions.append(pred)
    if idx % 1000 == 0 and idx != 0:
        print("Image " + str(idx) + "th")
        evaluate(predictions[:idx], labels[:idx])
