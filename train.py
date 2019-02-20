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


data_path = 'data/mydata'
vector_path_facenet = 'data/mydata_vecs_facenet'
vector_path_resnet = 'data/mydata_vecs'
if not os.path.exists(vector_path_facenet):
    os.makedirs(vector_path_facenet)
if not os.path.exists(vector_path_resnet):
    os.makedirs(vector_path_resnet)


def euclide_dist(emb1, emb2):
    distance = np.sqrt(np.sum(np.square(emb1 - emb2)))
    return distance


def cosine_dist(emb1, emb2):
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return similarity


def dump_data(vector_path):
    for person in os.listdir(data_path):
        person_images = os.path.join(data_path, person)
        for image_path in glob.glob(os.path.join(person_images, '*.jpg')):
            filename = image_path.split('/')[-1] + '.pkl'
            if os.path.exists(os.path.join(vector_path, person, filename)):
                continue
            image = utils.load_rgb_image(image_path)
            emb = utils.get_feature_vec(image, '')
            if not os.path.exists(os.path.join(vector_path, person)):
                os.makedirs(os.path.join(vector_path, person))
            pickle.dump(emb, open(os.path.join(vector_path, person, filename), 'wb'), pickle.HIGHEST_PROTOCOL)


def load_data(vector_path):
    data = []
    for person in sorted(os.listdir(data_path)):
        person_images = os.path.join(data_path, person)
        tmp = []
        for image_path in sorted(glob.glob(os.path.join(person_images, '*.jpg'))):
            filename = image_path.split('/')[-1] + '.pkl'
            try:
                emb = pickle.load(open(os.path.join(vector_path, person, filename), 'rb'))
                tmp.append(emb)
            except:
                pass
        data.append(tmp)
    return data


def get_train_data(data):
    X, y = [], []
    for i in range(1500):
        m = np.random.randint(len(data))
        n, k = np.random.randint(0, len(data[m]), size=2)
        euc_res = euclide_dist(data[m][n], data[m][k])
        cos_res = cosine_dist(data[m][n], data[m][k])
        X.append([euc_res, cos_res])
        y.append(1)
    for i in range(2000):
        n, k = np.random.randint(0, len(data), size=2)
        m = np.random.randint(len(data[n]))
        o = np.random.randint(len(data[k]))
        euc_res = euclide_dist(data[n][m], data[k][o])
        cos_res = cosine_dist(data[n][m], data[k][o])
        X.append([euc_res, cos_res])
        y.append(0)
    return shuffle(X, y)


# dump_data(vector_path_facenet)
# dump_data(vector_path_resnet)
# exit(0)
data_resnet = load_data(vector_path_resnet)
# data_facenet = load_data(vector_path_facenet)
X, y = get_train_data(data_resnet)
gs = svm.SVC(kernel='linear', probability=True, class_weight='balanced')
# gs = SGDClassifier()
# gs = svm.LinearSVC(class_weight='balanced')
# parameters = {'C': [0.5, 1]}
# gs = model_selection.GridSearchCV(gs, parameters, n_jobs=-1)
gs.fit(X, y)
# print(gs.best_params_)

pickle.dump(gs, open('data/model/gs.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
