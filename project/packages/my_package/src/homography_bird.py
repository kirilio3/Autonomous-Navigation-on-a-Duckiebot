#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image

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
        
        # Construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        # Publishers for rqt_image_view
        self.pub_original = rospy.Publisher(f"/{self._vehicle_name}/camera_reader/original", Image, queue_size=1)
        self.pub_edges = rospy.Publisher(f"/{self._vehicle_name}/camera_reader/edges", Image, queue_size=1)
        self.pub_birdseye = rospy.Publisher(f"/{self._vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
        
        # Define source points (to be tuned based on your camera's view)
        self.src_points = np.float32([
            [50, 500],  # bottom-left
            [620, 500],  # bottom-right
            [360, 180],  # top-right
            [250, 180]   # top-left
        ])
        
        # Define destination points for bird's-eye view (adjust as needed)
        self.dst_points = np.float32([
            [180, 500],  # bottom-left
            [440, 500],  # bottom-right
            [440, 0],    # top-right
            [180, 0]     # top-left
        ])
        
        # Calculate homography matrix
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def draw_points(self, img, points, color=(0, 0, 255), thickness=5):
        """Draw points and connect them with lines on the image"""
        img_with_points = img.copy()
        points = points.astype(int)
        
        # Draw points
        for point in points:
            cv2.circle(img_with_points, tuple(point), thickness, color, -1)
        
        # Draw connecting lines
        for i in range(len(points)):
            cv2.line(img_with_points, tuple(points[i]), tuple(points[(i + 1) % 4]), color, 2)
        
        return img_with_points

    def process_image(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply perspective transform
        birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
        return edges, birdseye

    def callback(self, msg):
        try:
            # Convert JPEG bytes to CV image
            image = self._bridge.compressed_imgmsg_to_cv2(msg)
            
            # Draw source points on original image
            image_with_points = self.draw_points(image, self.src_points, color=(0, 0, 255))  # Red points
            # image_with_points = self.draw_points(image, self.dst_points, color=(0, 0, 255))  # Red points
            
            # Process image for lane detection and transformation
            edges, birdseye = self.process_image(image)
            
            # Convert images to ROS Image messages
            original_msg = self._bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
            edges_msg = self._bridge.cv2_to_imgmsg(edges, encoding="mono8")
            birdseye_msg = self._bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            
            # Publish to topics
            self.pub_original.publish(original_msg)
            self.pub_edges.publish(edges_msg)
            self.pub_birdseye.publish(birdseye_msg)
            
        except Exception as e:
            self.log(f"Error processing image: {str(e)}", type='error')

    def on_shutdown(self):
        # Clean up on node shutdown
        super(CameraReaderNode, self).on_shutdown()

if __name__ == '__main__':
    # Create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # Keep spinning
    try:
        rospy.spin()
    except KeyboardInterrupt:
        node.on_shutdown()






# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage, Image

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
        
#         # Construct subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Publishers for rqt_image_view
#         self.pub_original = rospy.Publisher(f"/{self._vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self._vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self._vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
        
#         # Define source points (to be tuned based on your camera's view)
#         self.src_points = np.float32([
#             [100, 300],  # bottom-left
#             [540, 300],  # bottom-right
#             [420, 180],  # top-right
#             [220, 180]   # top-left
#         ])
        
#         # Define destination points for bird's-eye view (adjust as needed)
#         self.dst_points = np.float32([
#             [200, 400],  # bottom-left
#             [440, 400],  # bottom-right
#             [440, 0],    # top-right
#             [200, 0]     # top-left
#         ])
        
#         # Calculate homography matrix
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

#     def process_image(self, img):
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Edge detection using Canny
#         edges = cv2.Canny(blurred, 50, 150)
        
#         # Apply perspective transform
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
#         return edges, birdseye

#     def callback(self, msg):
#         try:
#             # Convert JPEG bytes to CV image
#             image = self._bridge.compressed_imgmsg_to_cv2(msg)
            
#             # Process image for lane detection and transformation
#             edges, birdseye = self.process_image(image)
            
#             # Convert images to ROS Image messages
#             original_msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
#             edges_msg = self._bridge.cv2_to_imgmsg(edges, encoding="mono8")
#             birdseye_msg = self._bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            
#             # Publish to topics
#             self.pub_original.publish(original_msg)
#             self.pub_edges.publish(edges_msg)
#             self.pub_birdseye.publish(birdseye_msg)
            
#         except Exception as e:
#             self.log(f"Error processing image: {str(e)}", type='error')

#     def on_shutdown(self):
#         # Clean up on node shutdown
#         super(CameraReaderNode, self).on_shutdown()

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         node.on_shutdown()




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
        
#         # Construct subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Define source points (to be tuned based on your camera's view)
#         self.src_points = np.float32([
#             [100, 300],  # bottom-left
#             [540, 300],  # bottom-right
#             [420, 180],  # top-right
#             [220, 180]   # top-left
#         ])
        
#         # Define destination points for bird's-eye view (adjust as needed)
#         self.dst_points = np.float32([
#             [200, 400],  # bottom-left
#             [440, 400],  # bottom-right
#             [440, 0],    # top-right
#             [200, 0]     # top-left
#         ])
        
#         # Calculate homography matrix
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

#     def process_image(self, img):
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Edge detection using Canny
#         edges = cv2.Canny(blurred, 50, 150)
        
#         # Apply perspective transform
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
#         return edges, birdseye

#     def callback(self, msg):
#         try:
#             # Convert JPEG bytes to CV image
#             image = self._bridge.compressed_imgmsg_to_cv2(msg)
            
#             # Process image for lane detection and transformation
#             edges, birdseye = self.process_image(image)
            
#             # Display original, edges, and bird's-eye view
#             cv2.imshow("Original", image)
#             cv2.imshow("Edges", edges)
#             cv2.imshow("Bird's Eye View", birdseye)
#             cv2.waitKey(1)
            
#         except Exception as e:
#             self.log(f"Error processing image: {str(e)}", type='error')

#     def on_shutdown(self):
#         # Clean up windows on node shutdown
#         cv2.destroyAllWindows()
#         super(CameraReaderNode, self).on_shutdown()

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         node.on_shutdown()