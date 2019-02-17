# CMC AI Contest

Group ID: a170!24

Members: Luong Tuan Dung, Bui Manh Thang, Bui Duy Tuan

## Installation

Environment: Ubuntu 16.04, Python 3.5+

Install dependencies
```
$ pip install tensorflow numpy opencv-python thrift imageio pandas scipy scikit-learn keras_vggface keras==2.2.0
```

## Usage

Open terminal at source folder
```
$ python thrift_server_resnet.py
```

Change path to data in config.py

SAMPLE_IMAGE_PATH is the path to the image of person who need searching

PUBLIC_TEST_PATH is the path to folder containing public test images


Then, in another termianl
```
$ python main.py
```

It should generate an output.csv file
