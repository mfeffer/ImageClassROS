#!/usr/bin/env python

import sys
import rospy
import roslib
import numpy as np
from scipy.ndimage import filters
from image_class.srv import *
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt

import gp

i = 0
image_taken = False

def classify_image_client(msg):
    try:
        sub.unregister()
        classify_image = rospy.ServiceProxy('classify_image_ros', ClassifyImage)
        resp1 = classify_image(msg)
        data = resp1.data
        entry_length = resp1.entry_length
        split_data = [data[i:i+entry_length] for i in range(0, len(data), entry_length)]
        print(split_data)
        # we can publish to whatever channel is necessary at this point
        for datum in split_data:
            xy = datum[0:2]
            classes = np.array(datum[3:])
            print(classes)
            print(xy)
            gp.new_image(classes,*xy)
        global image_taken
        image_taken = True
        # cm = plt.imshow(gp.get_image_map()[:,:,0],vmin=0,vmax=1)
        # plt.colorbar(cm)
        # plt.show("hold")
        return split_data

    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def usage():
    return "%s [x y]"%sys.argv[0]

global sub

if __name__ == "__main__":
    gp.setup()
    print "Requesting image"
    rospy.init_node('image_listener')
    image_topic = "/raspicam_node/image/compressed"
    sub = rospy.Subscriber(image_topic, CompressedImage, classify_image_client)
    while not rospy.core.is_shutdown() and not image_taken: #killable rospy spin()
        rospy.rostime.wallsleep(0.5)