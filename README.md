# CMC AI

## Usage

Run thrift server, 2 options: Resnet50 and InceptionResnet (facenet)
```
$ python thrift_server_resnet.py
```
or
```
$ python thrift_server_facenet.py
```
Then
```
$ python main.py
```

## Configuration

Edit configuration in config.py

Default model is Resnet. If you want to use Facenet, run thrift server facenet and edit config MODEL = 'facenet'

## Pretrained model
Resnet model is already in "keras_vggface" package

Facenet model is downloaded at https://github.com/davidsandberg/facenet
