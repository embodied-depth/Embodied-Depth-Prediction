#!/usr/bin/env python

import time
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import numpy as np
import skimage.transform

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cvb=CvBridge()

def callback(data):
    try:
        array = cvb.imgmsg_to_cv2(data)
        print(array.shape)
        #print(rospy.get_name(), "I heard %s"%(array.shape))
    except CvBridgeError as e:
        print(e)

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("floats", Image, callback)
    rospy.spin()


class DataListener:
    def __init__(self):
        self.rgb = []
        self.dep = []
        self.traj = None
        self.history_idx = 0

        rospy.Subscriber('data/rgb', Image, self.grab_rgb)
        rospy.Subscriber('data/depth', Image, self.grab_dep)
        rospy.Subscriber('data/traj', Image, self.grab_traj)

    def output_data(self):
        while self.traj is None:
            time.sleep(1)
            print('Waiting for the first batch data')

        out_len = min(len(self.rgb), len(self.dep), self.traj.shape[0])
        while out_len - self.history_idx < 5:
            time.sleep(1)
            print('Waiting for the batch data')
            out_len = min(len(self.rgb), len(self.dep), self.traj.shape[0])

        out_ = []
        for i in range(self.history_idx, out_len):
            out = {
                'color_sensor': self.rgb[i],
                'depth_sensor': self.dep[i] / 1000.,
                'Ext': self.traj[i]
            }
            out_.append(out)
        self.history_idx = out_len

        return out_, False


    def grab_rgb(self, data):
        try:
            array = cvb.imgmsg_to_cv2(data)
            self.rgb.append(array)
        except CvBridgeError as e:
            print(e)

    def grab_dep(self, data):
        try:
            array = cvb.imgmsg_to_cv2(data)
            depth = skimage.transform.resize(
                    array, (192, 640), order=0, preserve_range=True, mode='constant')
            self.dep.append(depth)
        except CvBridgeError as e:
            print(e)

    def grab_traj(self, data):
        try:
            array = cvb.imgmsg_to_cv2(data)
            if self.traj is  None:
                self.traj = array
            else:
                self.traj = np.concatenate([self.traj, array], axis=0)
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    listener()
