import os
import re
import numpy as np
from glob import glob
from datetime import datetime
import cv2


def get_files(directory, pattern):
    return np.asarray(sorted(glob(os.path.join(directory, pattern))))


def get_dt():
    return re.split(
        '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')


cap = cv2.VideoCapture(os.path.join())
fgbg = cv2.createBackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
