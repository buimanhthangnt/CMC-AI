# CMC AI Contest

Group ID: a170!24

Members: Luong Tuan Dung, Bui Manh Thang, Bui Duy Tuan

## Installation

Environment: Ubuntu 16.04, Python 3.5+ (virutal environment is recommended)

Install dependencies
```
$ pip install tensorflow numpy mxnet easydict opencv-python imageio pandas scipy scikit-learn scikit-image
```

## Usage

Change path to data in config.py

SAMPLE_IMAGE_PATH is the path to the image of person who need searching

PUBLIC_TEST_PATH is the path to folder containing public test images


Then, in another termianl
```
$ python main.py
```

It should generate an output.csv file
