#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        # Static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # Bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # Construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

        # Depth map publisher
        self.depth_pub = rospy.Publisher(f"/{self._vehicle_name}/depth_map", Image, queue_size=1)

    def callback(self, msg):
        # Convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Convert to grayscale (needed for depth computation)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a fake stereo pair (shifted image for depth approximation)
        shifted = np.roll(gray, shift=5, axis=1)

        # Initialize the StereoBM depth estimator
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
        depth_map = stereo.compute(gray, shifted)

        # Normalize depth map for visualization
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Publish the depth map
        depth_msg = self._bridge.cv2_to_imgmsg(depth_map, encoding="mono8")
        self.depth_pub.publish(depth_msg)

if __name__ == '__main__':
    # Create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # Keep spinning
    rospy.spin()