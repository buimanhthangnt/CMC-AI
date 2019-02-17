from imageio import imread
import os
import client_thrift
import utils
import config
from face_detection import detect_face
import numpy as np
import csv
from scipy import spatial


def get_target_face():
    rgb_image = imread(config.SAMPLE_IMAGE_PATH)
    ret, _ = detect_face(rgb_image)
    if len(ret) <= 0:
        raise Exception('no face in sample image')
    l, t, r, b = ret[0]
    return rgb_image[t:b,l:r]


def is_matched(emb1, emb2):
    similarity = 1 - spatial.distance.cosine(emb1, emb2)
    return abs(similarity) > 0.601


target_face = get_target_face()
target_embedding = np.array(client_thrift.get_emb_numpy([target_face])[0])

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
            l, t, r, b = bounding_box
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
        tmp.extend(bb)
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
