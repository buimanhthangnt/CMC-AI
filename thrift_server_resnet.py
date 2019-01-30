#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Face embedding Thrift server."""
import pickle
from keras.models import load_model
import config
import cv2
from gen_py.face_emb import FaceEmbedding
from keras_vggface import utils, VGGFace
import numpy as np
import tensorflow as tf
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport


class FaceEmbeddingHandler:
    """Request handler class."""
    def get_emb_numpy(self, numpy_imgs):
        images= []
        for pkl in numpy_imgs:
            img = pickle.loads(pkl)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype('float64')
            # img = img[:, :, ::-1]
            images.append(img)

        images = np.array(images)
        images = utils.preprocess_input(images, version=2)
        with graph.as_default():
            embeddings = vgg_features.predict(images)
        embeddings.shape = (-1, 2048)
        return embeddings


def main():
    print('Loading model...')
    global graph
    global vgg_features
    global image_size

    with tf.Graph().as_default() as graph:
        vgg_features = VGGFace(model='resnet50', include_top=False, pooling='avg')
        # vgg_features = load_model('resnet_model/resnet50.h5')
        image_size = 224
        handler = FaceEmbeddingHandler()
        processor = FaceEmbedding.Processor(handler)
        transport = TSocket.TServerSocket(
            host='0.0.0.0', port=config.SERVER_THRIFT_PORT)
        tfactory = TTransport.TBufferedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()
        server = TServer.TThreadedServer(
            processor, transport, tfactory, pfactory)
        print('READY')
        try:
            server.serve()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
