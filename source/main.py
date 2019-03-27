from imageio import imread
import os
import client_thrift
import utils
import config
from face_detection import detect_face
import numpy as np
import csv
import pickle
from scipy import spatial
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_target_face():
    rgb_image = imread(config.SAMPLE_IMAGE_PATH)
    ret, _ = detect_face(rgb_image)
    if len(ret) <= 0:
        raise Exception('no face in sample image')
    l, t, r, b = ret[0]
    return rgb_image


def cosine_dist(emb1, emb2):
    return spatial.distance.cosine(emb1, emb2)


def is_matched(emb1, emb2):
    cos = cosine_dist(emb1, emb2)
    # print(cos)
    return cos <= 0.7


target_face = get_target_face()
target_embedding = utils.get_feature_vec(target_face)

predictions = []
print("Running")
for idx, fname in enumerate(sorted(os.listdir(config.PUBLIC_TEST_PATH))):
    image_path = os.path.join(config.PUBLIC_TEST_PATH, fname)
    image = utils.load_rgb_image(image_path)
    pred, bb = 0, None
    if image is not None:
        bbs, _ = detect_face(image.copy())
        if len(bbs) == 0: 
            print("No face found in: ", image_path)
        for idx2, bounding_box in enumerate(bbs):
            l, t, r, b = utils.add_padding(image, bounding_box, (0.2, 0.2))
            face = image[t:b,l:r]
            face_emb = utils.get_feature_vec(face, fname + str(idx2))
            if is_matched(target_embedding, face_emb):
                pred = 1
                bb = bounding_box
    fname = fname.rsplit('.')[0]
    if pred == 0:
        predictions.append([fname, 0, 0, 0, 0, pred])
    else:
        tmp = [fname]
        tmp.extend(utils.add_padding(image, bb, (0.12, 0.08)))
        tmp.append(pred)
        predictions.append(tmp)
    if idx % 100 == 0 and idx != 0:
        print("Image " + str(idx) + "th")
    # if idx > 100:
    #     break

csv_content = [["image", "x1", "y1", "x2", "y2", "result"]]
csv_content.extend(predictions)
with open('output.csv', 'w') as fn:
    csv.writer(fn).writerows(csv_content)
