from insightface.deploy import face_model
import argparse
import cv2
import sys
import numpy as np


model = face_model.FaceModel()


def get_emb(bgr_img):
    img = model.get_input(bgr_img)
    # if img is None: return None
    return model.get_feature(img)

