#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # Static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
        # CV bridge
        self._bridge = CvBridge()
        
        # Optical flow parameters
        self.prev_gray = None
        self.hsv = None
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(winSize=(15, 15),
                            maxLevel=2,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Feature parameters for good features to track
        self.feature_params = dict(maxCorners=10,
                                 qualityLevel=0.9,
                                 minDistance=1,
                                 blockSize=32)
        
        # Publisher for optical flow image
        self.pub_flow = rospy.Publisher(f"/{self._vehicle_name}/optical_flow/image/compressed",
                                      CompressedImage,
                                      queue_size=1)
        
        # Subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        # Initialize point tracking
        self.prev_points = None

    def callback(self, msg):
        # Convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Calculate optical flow using Lucas-Kanade method
            if self.prev_points is not None:
                next_points, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None, **self.lk_params)
                
                # Select good points
                good_new = next_points[status == 1]
                good_old = self.prev_points[status == 1]
                
                # Create visualization
                flow_vis = image.copy()
                
                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # Draw line between old and new points
                    flow_vis = cv2.line(flow_vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    # Draw current point
                    flow_vis = cv2.circle(flow_vis, (int(a), int(b)), 5, (0, 0, 255), -1)
                
                # Detect potential obstacles (based on flow magnitude)
                flow_magnitude = np.sqrt((good_new - good_old) ** 2).sum(axis=1)
                if np.any(flow_magnitude > 20):  # Adjust threshold as needed
                    rospy.logwarn("Potential obstacle detected!")
                
                # Convert to compressed image message and publish
                msg_out = self._bridge.cv2_to_compressed_imgmsg(flow_vis)
                msg_out.header = msg.header
                self.pub_flow.publish(msg_out)
            
            # Update points to track
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        # Optional: Add small delay
        cv2.waitKey(1)

if __name__ == '__main__':
    # Create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # Keep spinning
    rospy.spin()





# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage, Image
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class OpticalFlowNode(DTROS):
#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(OpticalFlowNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot')  # Default to 'duckiebot' if not set
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
        
#         # Publisher for the processed image (to display in rqt_image_view)
#         self.pub = rospy.Publisher(f"/{self._vehicle_name}/optical_flow_image", Image, queue_size=1)
        
#         # Subscriber to the Duckiebot camera
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Initialize previous frame for optical flow
#         self.prev_gray = None
        
#         # Block size for optical flow visualization
#         self.block_size = 32
        
#         # Video writer (optional, for saving output)
#         self.frame_width = 640  # Adjust based on Duckiebot camera resolution
#         self.frame_height = 480
#         self.out = cv2.VideoWriter('duckiebot_optical_flow.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
#                                   (self.frame_width, self.frame_height))

#     def callback(self, msg):
#         # Convert compressed ROS image to OpenCV format
#         img = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Convert to grayscale for optical flow
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         if self.prev_gray is not None:
#             # Compute optical flow using Farneback method
#             flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
#             # Visualize flow with arrows
#             h, w = flow.shape[:2]
#             for y in range(0, h, self.block_size):
#                 for x in range(0, w, self.block_size):
#                     fx, fy = flow[y, x]
#                     cv2.arrowedLine(img, (x, y), (int(x + fx), int(y + fy)), (0, 0, 255), 2)
            
#             # Publish the processed image to ROS topic
#             processed_img_msg = self._bridge.cv2_to_imgmsg(img, encoding="bgr8")
#             self.pub.publish(processed_img_msg)
            
#             # Optionally write to video file
#             self.out.write(img)
        
#         # Update previous frame
#         self.prev_gray = gray.copy()

#     def on_shutdown(self):
#         # Release video writer and cleanup
#         self.out.release()
#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     # Create the node
#     node = OpticalFlowNode(node_name='optical_flow_node')
#     # Keep spinning
#     rospy.spin()





# #!/usr/bin/env python3

# import os
# import rospy
# import cv2
# import numpy as np
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge

# class CameraReaderNode(DTROS):
#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
#         # CV bridge
#         self._bridge = CvBridge()
        
#         # Optical flow parameters
#         self.prev_gray = None
#         self.hsv = None
        
#         # Lucas-Kanade optical flow parameters
#         self.lk_params = dict(winSize=(15, 15),
#                             maxLevel=2,
#                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
#         # Feature parameters for good features to track
#         self.feature_params = dict(maxCorners=100,
#                                  qualityLevel=0.3,
#                                  minDistance=7,
#                                  blockSize=7)
        
#         # Publisher for optical flow image
#         self.pub_flow = rospy.Publisher(f"/{self._vehicle_name}/optical_flow/image/compressed",
#                                       CompressedImage,
#                                       queue_size=1)
        
#         # Subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Initialize point tracking
#         self.prev_points = None

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         if self.prev_gray is not None:
#             # Calculate optical flow using Lucas-Kanade method
#             if self.prev_points is not None:
#                 next_points, status, err = cv2.calcOpticalFlowPyrLK(
#                     self.prev_gray, gray, self.prev_points, None, **self.lk_params)
                
#                 # Select good points
#                 good_new = next_points[status == 1]
#                 good_old = self.prev_points[status == 1]
                
#                 # Create visualization
#                 flow_vis = image.copy()
                
#                 # Draw the tracks
#                 for i, (new, old) in enumerate(zip(good_new, good_old)):
#                     a, b = new.ravel()
#                     c, d = old.ravel()
#                     # Draw line between old and new points
#                     flow_vis = cv2.line(flow_vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#                     # Draw current point
#                     flow_vis = cv2.circle(flow_vis, (int(a), int(b)), 5, (0, 0, 255), -1)
                
#                 # Detect potential obstacles (based on flow magnitude)
#                 flow_magnitude = np.sqrt((good_new - good_old) ** 2).sum(axis=1)
#                 if np.any(flow_magnitude > 20):  # Adjust threshold as needed
#                     rospy.logwarn("Potential obstacle detected!")
                
#                 # Convert to compressed image message and publish
#                 msg_out = self._bridge.cv2_to_compressed_imgmsg(flow_vis)
#                 msg_out.header = msg.header
#                 self.pub_flow.publish(msg_out)
            
#             # Update points to track
#             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
#         # Update previous frame
#         self.prev_gray = gray.copy()
        
#         # Optional: Add small delay
#         cv2.waitKey(1)

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     rospy.spin()