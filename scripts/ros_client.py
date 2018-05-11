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

bridge = CvBridge()

i = 0

def classify_image_client(msg):
#    rospy.wait_for_service('classify_image_ros')
#    remember to fix comments regarding classification
    # try:
    #     sub.unregister()
    #     classify_image = rospy.ServiceProxy('classify_image_ros', ClassifyImage)
    #     resp1 = classify_image(msg)
    #     data = resp1.data
    #     entry_length = resp1.entry_length
    #     split_data = [data[i:i+entry_length] for i in range(0, len(data), entry_length)]
    #     print(data)
    #     print(split_data)
    #     return split_data
    # except rospy.ServiceException, e:
    #     print "Service call failed: %s"%e


    global i
    print(i)
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, 1)
    #rgb_image = Image.fromarray(image_np[...,::-1])
    cv2.imwrite('/home/michael/ros_images/ros_'+str(i)+'.jpg',image_np)
    i += 1

def usage():
    return "%s [x y]"%sys.argv[0]

global sub

if __name__ == "__main__":

    print "Requesting image"
    rospy.init_node('image_listener')
    image_topic = "/raspicam_node/image/compressed"
    sub = rospy.Subscriber(image_topic, CompressedImage, classify_image_client)
    rospy.spin()
    #print "%s"%(classify_image_client())
