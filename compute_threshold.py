import os
import cv2
import pickle
import glob
import numpy as np
import client_thrift


path = '/root/data/face_label_good2'
vects = []
count = 0
for person in os.listdir(path):
    p_emb = []
    for filepath in glob.glob(os.path.join(path, person, '*.jpg')):
        # emb = pickle.load(open(filepath, 'rb'))
        face = cv2.imread(filepath)
        emb = client_thrift.get_emb_numpy([face])[0]
        p_emb.append(emb)
        count += 1
        if count % 20 == 0:
            print(count)
    vects.append(np.array(p_emb))


distance = []
for i in range(10000):
    j = np.random.randint(len(vects))
    m, n = np.random.randint(0, vects[j].shape[0], size=2)
    dist = np.sqrt(np.sum(np.square(vects[j][m] - vects[j][n])))
    distance.append(dist)


print(np.std(np.array(distance)))
print(np.mean(np.array(distance)))


distance = []
for i in range(10000):
    m, n = np.random.randint(0, len(vects), size=2)
    j = np.random.randint(vects[m].shape[0])
    k = np.random.randint(vects[n].shape[0])
    dist = np.sqrt(np.sum(np.square(vects[m][j] - vects[n][k])))
    distance.append(dist)


print(np.std(np.array(distance)))
print(np.mean(np.array(distance)))
