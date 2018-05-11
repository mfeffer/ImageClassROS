#!/usr/bin/env python

import rospy
import api
import roslib
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import keras as K
import tensorflow as tf

from image_class.srv import *

def classify_image(req):
    np_arr = np.fromstring(req.image.data, np.uint8)
    image_np = cv2.imdecode(np_arr, 1)
    rgb_image = Image.fromarray(image_np[...,::-1])
    cv2.imwrite('/home/michael/camera_image.jpeg',image_np)
    data = api.classify_image(rgb_image, img_save=True)
    return ClassifyImageResponse(data, 2, 7)
    
def classify_image_server():
    rospy.init_node('classify_image_server')
    s = rospy.Service('classify_image_ros', ClassifyImage, classify_image)
    print "Ready to classify image."
    rospy.spin()

if __name__ == "__main__":
    classify_image_server()
