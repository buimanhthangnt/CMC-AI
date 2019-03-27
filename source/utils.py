import os
import re
import cv2
import config
import pickle
import numpy as np
import client_thrift
import tensorflow as tf
from insightface.deploy.feature_extraction import get_emb


def load_rgb_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image error: ", image_path)
        return image
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_feature_vec(face, fname=None):
    # if not os.path.exists(config.VECTORS_PATH):
    #     os.makedirs(config.VECTORS_PATH)
    # fpath = os.path.join(config.VECTORS_PATH, fname + '.pkl')
    # if os.path.exists(fpath):
    #     face_emb = pickle.load(open(fpath, 'rb'))
    # else:
    face_emb = get_emb(face[:,:,::-1])
    # face_emb = np.array(client_thrift.get_emb_numpy([face])[0])
        # pickle.dump(face_emb, open(fpath, 'wb'), pickle.HIGHEST_PROTOCOL)
    return face_emb


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def add_padding(img, locations, offsets, padding=False):
    l, t, r, b = locations
    xw1 = max(int(l - offsets[0] * (r - l)), 0)
    yw1 = max(int(t - offsets[1] * (b - t)), 0)
    xw2 = min(int(r + offsets[0] * (r - l)), img.shape[1] - 1)
    yw2 = min(int(b + offsets[1] * (b - t)), img.shape[0] - 1)
    return xw1, yw1, xw2, yw2
