#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
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
        
        # Publishers for rqt_image_view
        self.pub_original = rospy.Publisher(
            f"/{self._vehicle_name}/camera_reader/original/compressed",
            CompressedImage,
            queue_size=1
        )
        self.pub_matches = rospy.Publisher(
            f"/{self._vehicle_name}/camera_reader/matches/compressed",
            CompressedImage,
            queue_size=1
        )
        
        # Construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        # Initialize variables for homography
        self.prev_image = None
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Counter for processing every nth frame
        self.frame_count = 0
        self.process_every_n_frames = 2
        
        # Threshold for detecting significant movement (in pixels)
        self.movement_threshold_2 = 5.0  # Adjust this value based on your setup
        self.movement_threshold = 1.0

    def callback(self, msg):
        # Convert JPEG bytes to CV image
        curr_image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # Prepare and publish original image
        original_msg = self._bridge.cv2_to_compressed_imgmsg(curr_image)
        original_msg.header = msg.header  # Preserve timestamp and frame_id
        self.pub_original.publish(original_msg)
        
        # Process every nth frame
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames == 0 and self.prev_image is not None:
            # Convert to grayscale for feature detection
            curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
            
            # Detect ORB features and compute descriptors
            kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
            kp2, des2 = self.orb.detectAndCompute(curr_gray, None)
            
            if des1 is not None and des2 is not None:
                # Match descriptors
                matches = self.bf.match(des1, des2)
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Extract matched points
                if len(matches) >= 4:  # Minimum points needed for homography
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Calculate homography
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    # Draw matches
                    match_img = cv2.drawMatches(
                        prev_gray, kp1, curr_gray, kp2, matches[:50],
                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    
                    # Convert and publish match image
                    match_msg = self._bridge.cv2_to_compressed_imgmsg(match_img)
                    match_msg.header = msg.header
                    self.pub_matches.publish(match_msg)
                    
                    # Log homography matrix
                    # self.loginfo(f"Homography matrix:\n{H}")
                    
                    # Analyze translation from homography
                    tx = H[0, 2]  # Horizontal translation
                    ty = H[1, 2]  # Vertical translation
                    
                    # Determine movement direction
                    if abs(tx) < self.movement_threshold_2 and abs(ty) < self.movement_threshold:
                        # Stationary: no output
                        pass
                    elif abs(tx) >= self.movement_threshold_2:
                        if tx < 0:
                            self.loginfo(f"Moved right by {tx:.2f} pixels")
                        else:
                            self.loginfo(f"Moved left by {abs(tx):.2f} pixels")
                    elif abs(ty) >= self.movement_threshold:
                        if ty < 0:
                            self.loginfo(f"Moved forward by {ty:.2f} pixels")
                        else:
                            self.loginfo(f"Moved backward by {abs(ty):.2f} pixels")

        # Update previous image
        self.prev_image = curr_image.copy()

if __name__ == '__main__':
    # Create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # Keep spinning
    rospy.spin()


# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage

# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class CameraReaderNode(DTROS):

#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
        
#         # Publishers for rqt_image_view
#         self.pub_original = rospy.Publisher(
#             f"/{self._vehicle_name}/camera_reader/original/compressed",
#             CompressedImage,
#             queue_size=1
#         )
#         self.pub_matches = rospy.Publisher(
#             f"/{self._vehicle_name}/camera_reader/matches/compressed",
#             CompressedImage,
#             queue_size=1
#         )
        
#         # Construct subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Initialize variables for homography
#         self.prev_image = None
#         self.orb = cv2.ORB_create(nfeatures=500)
#         self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
#         # Counter for processing every nth frame
#         self.frame_count = 0
#         self.process_every_n_frames = 5

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         curr_image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Prepare and publish original image
#         original_msg = self._bridge.cv2_to_compressed_imgmsg(curr_image)
#         original_msg.header = msg.header  # Preserve timestamp and frame_id
#         self.pub_original.publish(original_msg)
        
#         # Process every nth frame
#         self.frame_count += 1
#         if self.frame_count % self.process_every_n_frames == 0 and self.prev_image is not None:
#             # Convert to grayscale for feature detection
#             curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
#             prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
            
#             # Detect ORB features and compute descriptors
#             kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
#             kp2, des2 = self.orb.detectAndCompute(curr_gray, None)
            
#             if des1 is not None and des2 is not None:
#                 # Match descriptors
#                 matches = self.bf.match(des1, des2)
#                 # Sort matches by distance
#                 matches = sorted(matches, key=lambda x: x.distance)
                
#                 # Extract matched points
#                 if len(matches) >= 4:  # Minimum points needed for homography
#                     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#                     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
#                     # Calculate homography
#                     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
#                     # Draw matches
#                     match_img = cv2.drawMatches(
#                         prev_gray, kp1, curr_gray, kp2, matches[:50],
#                         None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#                     )
                    
#                     # Convert and publish match image
#                     match_msg = self._bridge.cv2_to_compressed_imgmsg(match_img)
#                     match_msg.header = msg.header
#                     self.pub_matches.publish(match_msg)
                    
#                     # Log homography matrix
#                     self.loginfo(f"Homography matrix:\n{H}")
        
#         # Update previous image
#         self.prev_image = curr_image.copy()

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     rospy.spin()