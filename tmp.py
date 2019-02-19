import cv2
import os


path = "data/public_test"
import sys
fname = sys.argv[1]
fpath = os.path.join(path, fname)
image = cv2.imread(fpath)
cv2.imshow("test", image)
cv2.waitKey(0)
