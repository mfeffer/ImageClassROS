#!/usr/bin/env python

import api
from PIL import Image
import numpy as np
from keras.models import load_model
import keras as K
import tensorflow as tf
import cv2

path_to_image = "/home/michael/ROSClassify/new_test/imgDistance2.jpg"
cv_img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
cv_img = Image.fromarray(cv_img[...,::-1])
img = Image.open(path_to_image)
np.testing.assert_array_equal(img, cv_img)
img = img.resize((640, 480))
classifs = api.classify_image(img, img_save=True) # (mix - 31) (sand - 57, 53) (none - 58) (coral - 38) (cool result - 79)
split_data = [classifs[i:i+7] for i in range(0, len(classifs), 7)]
print(split_data)
