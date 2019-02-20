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


def get_target_face(target_face_path):
    rgb_image = imread(target_face_path)
    ret, _ = detect_face(rgb_image)
    if len(ret) <= 0:
        raise Exception('no face in sample image')
    l, t, r, b = ret[0]
    return rgb_image[t:b,l:r]


def get_annotation_data(anno_path):
    data = pd.read_csv(anno_path).values
    labels = data[:,-1]
    imgs = data[:,0]
    coords = data[:,1:-1]
    return labels, imgs, coords


def euclide_dist(emb1, emb2):
    distance = np.linalg.norm(emb1 - emb2)
    return distance


def cosine_dist(emb1, emb2):
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return similarity


def is_matched_cosine(emb1, emb2):
    similarity = cosine_dist(emb1, emb2)
    return abs(similarity) > 0.601


def is_matched_euclide(emb1, emb2):
    return euclide_dist(emb1, emb2) < 108


def is_matched_svm(emb_target, emb):
    euc = euclide_dist(emb_target, emb)
    cos = cosine_dist(emb_target, emb)
    gs = pickle.load(open('data/best_model/gs.pkl', 'rb'))
    pred = gs.predict_proba([[euc, cos]])[0]
    return pred[1] >= 0.575


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
        precision = 0
        recall = 0
        F1_score = 0
        pass
    return precision, recall, F1_score


test_features = pkl.load(open("data/publictest_resnetfeatures.p","rb"))

labels, imgs, coords = get_annotation_data(config.ANNOTATION_PATH)
# for idx, fname in enumerate(imgs):
#     if fname[:3] != 'PQH': continue
#     image = cv2.imread(os.path.join(config.PUBLIC_TEST_PATH, fname + '.jpg'))
#     ret, _ = detect_face(image.copy()[:,:,::-1])
#     for bb in ret:
#         l, t, r, b = utils.add_padding(image, bb, (0.12, 0.08), padding=True)
#         cv2.rectangle(image, (l,t), (r,b), (255,0,0), 2)
#     l, t, r, b = coords[idx]
#     cv2.rectangle(image, (l,t), (r,b), (0,255,0), 2)
#     cv2.imshow("test", image)
#     cv2.waitKey(0)

# exit()

real_imgs = ['_'.join(x.split('_')[:-1]) for x in imgs]
unique_people = list(set(real_imgs))
PS, RS, FS = [], [], []
for person in sorted(unique_people):
    fnames = []
    labels = np.zeros((len(imgs),))
    count = 0
    for idx, img in enumerate(imgs):
        if person in img:
            fnames.append(img)
            labels[idx] = 1
            count += 1
    if count < 30:
        continue
    print(person)
    target_face = get_target_face(os.path.join(config.PUBLIC_TEST_PATH, fnames[2] + '.jpg'))
    target_embedding = np.array(client_thrift.get_emb_numpy([target_face])[0])
    
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
    P, R, F1 = evaluate(predictions[:idx], labels[:idx])
    PS.append(P)
    RS.append(R)
    FS.append(F1)

print("Final result")
print("Precision: ", np.mean(PS))
print("Recall:", np.mean(RS))
print("F1: ", np.mean(FS))
