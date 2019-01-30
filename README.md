# CMC AI

## Usage

Download dataset at https://drive.google.com/drive/folders/1vfOy2jhDxo6oCpsgzmuMCNsch0FbgLah

Run thrift server, 2 options: Resnet50 and InceptionResnet (facenet)
```
$ python thrift_server_resnet.py
```
```
$ python thrift_server_facenet.py
```
Run evaluation
```
$ python main.py
```


## Configuration

Edit configuration in config.py



## Pretrained model
Resnet model is already in "keras_vggface" package

Facenet model is downloaded at https://github.com/davidsandberg/facenet
