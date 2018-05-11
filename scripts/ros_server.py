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

#dummy function until real classification gets implemented
#def classify_image(req):
def classify_image(req):
    #print(req.a)
    np_arr = np.fromstring(req.image.data, np.uint8)
    image_np = cv2.imdecode(np_arr, 1)
    #rgb_image = Image.fromarray(np.roll(image_np, 1, axis=-1))
    rgb_image = Image.fromarray(image_np[...,::-1])
    cv2.imwrite('/home/michael/camera_image.jpeg',image_np)
    #data = api.classify_image(rgb_image, model, graph, img_save=True)
    data = api.classify_image(rgb_image, img_save=True)
    return ClassifyImageResponse(data, 2, 7)
    #return ClassifyImageResponse(1, 2., 2., 3., 4., 3.)

def classify_image_server():
    rospy.init_node('classify_image_server')
    #global model
    #model = load_model('grand_challenge_trained_new.h5')
    #global graph
    #graph = K.backend.get_session().graph
    s = rospy.Service('classify_image_ros', ClassifyImage, classify_image)
    print "Ready to classify image."
    rospy.spin()

if __name__ == "__main__":
    classify_image_server()
