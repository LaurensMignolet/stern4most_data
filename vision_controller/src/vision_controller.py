#!/usr/bin/python

import rospy
import time

from sensor_msgs.msg import Image
from threading import Lock

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

from geometry_msgs.msg import Twist
import datetime
GUI_UPDATE_PERIOD = 0.10  # Seconds


class VisionDisplay:

    def __init__(self):
        self.running = True
        self.subVideo   = rospy.Subscriber('/camera/rgb/image_raw', Image, self.callback_image_raw)
        self.subVideo   = rospy.Subscriber('/cmd_vel', Twist, self.callback_controls)

        self.bridge = CvBridge()

        self.image = None
        self.imageLock = Lock()
        self.latest_image = None

        self.bound_low = np.array([0, 0, 0])
        self.bound_up  = np.array([0, 0, 0])

        self.statusMessage = ''

        self.connected = False

        self.redrawTimer = rospy.Timer(rospy.Duration(GUI_UPDATE_PERIOD), self.callback_redraw)

    def is_running(self):
        return self.running

    def convert_ros_to_opencv(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            return cv_image
        except CvBridgeError as error:
            raise Exception("Failed to convert to OpenCV image")

    def callback_redraw(self, event):
        if self.running == True and self.image is not None:
            self.imageLock.acquire()
            try:
                # Convert the captured frame from ROS to OpenCV.
                image_cv = self.convert_ros_to_opencv(self.image)
            finally:
                self.imageLock.release()

            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            img = cv2.resize(image_cv,(360,480))
            cv2.imshow("Image", img)
            self.latest_image = img


            key = cv2.waitKey(5)
            if key == 27: # Esc key top stop
                cv2.destroyAllWindows()
                self.running = False

    def callback_trackbars(self, value):
        h_low = cv2.getTrackbarPos('Hue lower bound:', 'Mask')
        h_up  = cv2.getTrackbarPos('Hue upper bound:', 'Mask')
        s_low = 0
        s_up  = 0
        v_low = 0
        v_up  = 0

        self.bound_low = np.array([h_low, s_low, v_low], np.uint8)
        self.bound_up = np.array([h_up, s_up, v_up], np.uint8)

    def callback_controls(self, data):
        z = data.angular.z
        if(z<0.01 and z > -0.01):
            z = 0
        print(z)
        cv2.imshow("latest_image ", self.latest_image)
        #print(datetime.datetime.now())
        filepath = "./images/" + str(z) + "_" + str(datetime.datetime.now().time()) +".jpg"
        cv2.imwrite(filepath , self.latest_image );
        print(filepath)

    def callback_image_raw(self, data):
        self.imageLock.acquire()
        try:
            self.image = data
        finally:
            self.imageLock.release()


if __name__=='__main__':
    rospy.init_node('vision_controller')

    display = VisionDisplay()

    while display.is_running():
        time.sleep(5)
