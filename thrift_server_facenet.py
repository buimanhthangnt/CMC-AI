#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Face embedding Thrift server."""
import pickle
import config
import cv2
import utils
from gen_py.face_emb import FaceEmbedding
import tensorflow as tf
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport


class FaceEmbeddingHandler:
    """Request handler class."""

    def get_emb_numpy(self, numpy_imgs):
        """Handler function for list of pickled numpy images."""
        images = []
        for pkl in numpy_imgs:
            npimg = pickle.loads(pkl)
            img = cv2.resize(npimg, (image_size, image_size))
            # img = img[:, :, ::-1]
            img = utils.prewhiten(img)
            images.append(img)
        feed_dict = {images_placeholder: images,
                     phase_train_placeholder: False}
        vecs = sess.run(embeddings, feed_dict=feed_dict)
        return vecs


def main():
    print('Loading model...')
    global embeddings
    global sess
    global image_size
    global embedding_size
    global images_placeholder
    global phase_train_placeholder
    model_dir = 'facenet_model/old'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        utils.load_model(model_dir)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        image_size = images_placeholder.get_shape()[1]
        embedding_size = embeddings.get_shape()[1]

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
