#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge
import numpy as np
import signal
import sys

class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()
        
        # Camera parameters (calibrate these)
        self.fx = 300.0
        self.fy = 300.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Publishers
        self.pub_cloud = rospy.Publisher(f"/{self._vehicle_name}/point_cloud", PointCloud2, queue_size=10)
        self.pub_image = rospy.Publisher(f"/{self._vehicle_name}/camera_debug/compressed", CompressedImage, queue_size=10)
        
        # Subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        # Signal handler for saving on Ctrl+C
        signal.signal(signal.SIGINT, self.save_and_exit)
        self.cloud_msg = None  # Store the latest point cloud

    def callback(self, msg):
        # Convert to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        self.pub_image.publish(msg)
        
        # Simple depth estimation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        depth = (gray / 255.0) * 5.0
        
        # Downsample
        height, width = image.shape[:2]
        image = cv2.resize(image, (width//2, height//2))
        depth = cv2.resize(depth, (width//2, height//2))
        height, width = image.shape[:2]
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        x = (u - self.cx/2) * depth / self.fx
        y = (v - self.cy/2) * depth / self.fy
        z = depth
        
        # Stack into point cloud
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        colors = image.reshape(-1, 3)
        
        # Create point cloud data
        cloud_data = []
        for point, color in zip(points, colors):
            x, y, z = point
            r, g, b = color
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            cloud_data.append([x, y, z, rgb])
        
        # Define fields
        fields = [
            pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
        ]
        
        # Create PointCloud2 message
        header = msg.header
        header.frame_id = f"{self._vehicle_name}/camera_frame"
        self.cloud_msg = pc2.create_cloud(header, fields, cloud_data)  # Store for saving
        
        # Publish
        self.pub_cloud.publish(self.cloud_msg)

    def save_and_exit(self, sig, frame):
        if self.cloud_msg:
            # Convert PointCloud2 to PCD format
            points = list(pc2.read_points(self.cloud_msg, field_names=("x", "y", "z", "rgb")))
            with open(f"/tmp/{self._vehicle_name}_pointcloud.pcd", "w") as f:
                f.write("# .PCD v0.7 - Point Cloud Data file format\n")
                f.write("VERSION 0.7\n")
                f.write("FIELDS x y z rgb\n")
                f.write("SIZE 4 4 4 4\n")
                f.write("TYPE F F F U\n")
                f.write("COUNT 1 1 1 1\n")
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                for p in points:
                    x, y, z, rgb = p
                    f.write(f"{x} {y} {z} {rgb}\n")
            rospy.loginfo(f"Saved point cloud to /tmp/{self._vehicle_name}_pointcloud.pcd")
        rospy.signal_shutdown("User requested shutdown")

if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()





# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage, PointCloud2
# import sensor_msgs.point_cloud2 as pc2
# import cv2
# from cv_bridge import CvBridge
# import numpy as np

# class CameraReaderNode(DTROS):
#     def __init__(self, node_name):
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._bridge = CvBridge()
        
#         # Camera parameters (calibrate these)
#         self.fx = 300.0
#         self.fy = 300.0
#         self.cx = 320.0
#         self.cy = 240.0
        
#         # Publishers
#         self.pub_cloud = rospy.Publisher(f"/{self._vehicle_name}/point_cloud", PointCloud2, queue_size=10)
#         self.pub_image = rospy.Publisher(f"/{self._vehicle_name}/camera_debug/compressed", CompressedImage, queue_size=10)
        
#         # Subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

#     def callback(self, msg):
#         # Convert to CV image
#         image = self._bridge.compressed_imgmsg_to_cv2(msg)
#         self.pub_image.publish(msg)
        
#         # Simple depth estimation: use grayscale intensity as a proxy (0-5 meters)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         depth = (gray / 255.0) * 5.0  # Scale intensity (0-255) to depth (0-5 meters)
        
#         # Downsample for performance (optional)
#         height, width = image.shape[:2]
#         image = cv2.resize(image, (width//2, height//2))
#         depth = cv2.resize(depth, (width//2, height//2))
#         height, width = image.shape[:2]
        
#         # Create coordinate grids
#         u, v = np.meshgrid(np.arange(width), np.arange(height))
        
#         # Convert to 3D coordinates
#         x = (u - self.cx/2) * depth / self.fx  # Adjust cx, cy for downsampled image
#         y = (v - self.cy/2) * depth / self.fy
#         z = depth
        
#         # Stack into point cloud
#         points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#         colors = image.reshape(-1, 3)
        
#         # Create point cloud data
#         cloud_data = []
#         for point, color in zip(points, colors):
#             x, y, z = point
#             r, g, b = color
#             rgb = (int(r) << 16) | (int(g) << 8) | int(b)
#             cloud_data.append([x, y, z, rgb])
        
#         # Define fields
#         fields = [
#             pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
#             pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
#             pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
#             pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
#         ]
        
#         # Create and publish PointCloud2
#         header = msg.header
#         header.frame_id = f"{self._vehicle_name}/camera_frame"
#         cloud_msg = pc2.create_cloud(header, fields, cloud_data)
#         self.pub_cloud.publish(cloud_msg)

# if __name__ == '__main__':
#     node = CameraReaderNode(node_name='camera_reader_node')
#     rospy.spin()


# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage, PointCloud2
# import sensor_msgs.point_cloud2 as pc2
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# from std_msgs.msg import Header

# class CameraReaderNode(DTROS):
#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
        
#         # Camera parameters (calibrate these)
#         self.fx = 300.0  # focal length x (example value)
#         self.fy = 300.0  # focal length y (example value)
#         self.cx = 320.0  # optical center x (example value)
#         self.cy = 240.0  # optical center y (example value)
        
#         # Publisher for point cloud
#         self.pub_cloud = rospy.Publisher(f"/{self._vehicle_name}/point_cloud", PointCloud2, queue_size=10)
        
#         # Publisher for raw image (for rqt_image_view)
#         self.pub_image = rospy.Publisher(f"/{self._vehicle_name}/camera_debug/compressed", CompressedImage, queue_size=10)
        
#         # Subscriber for camera
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Republish the raw image for rqt_image_view
#         self.pub_image.publish(msg)
        
#         # Dummy depth map (replace with real depth data if available)
#         height, width = image.shape[:2]
#         # In callback, replace the depth line:
#         depth = np.linspace(0.5, 2.0, height * width).reshape(height, width).astype(np.float32)  # Varies from 0.5m to 2m
#         # depth = np.ones((height, width), dtype=np.float32) * 1.0  # 1 meter depth
        
#         # Create coordinate grids
#         u, v = np.meshgrid(np.arange(width), np.arange(height))
        
#         # Convert to 3D coordinates
#         x = (u - self.cx) * depth / self.fx
#         y = (v - self.cy) * depth / self.fy
#         z = depth
        
#         # Stack into point cloud
#         points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        
#         # Add RGB colors from the image
#         colors = image.reshape(-1, 3)  # Shape: (n_points, 3) with [R, G, B]
        
#         # Create point cloud data with XYZRGB
#         cloud_data = []
#         for point, color in zip(points, colors):
#             x, y, z = point
#             r, g, b = color
#             rgb = (int(r) << 16) | (int(g) << 8) | int(b)
#             cloud_data.append([x, y, z, rgb])
        
#         # Define fields for PointCloud2 (x, y, z, rgb)
#         fields = [
#             pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
#             pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
#             pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
#             pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
#         ]
        
#         # Create PointCloud2 message
#         header = msg.header
#         header.frame_id = f"{self._vehicle_name}/camera_frame"
        
#         cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        
#         # Publish the point cloud
#         self.pub_cloud.publish(cloud_msg)
        
#         # Optional: Display the original image locally
#         # cv2.imshow("camera", image)
#         # cv2.waitKey(1)

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     rospy.spin()






# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# from collections import deque
# import open3d as o3d  # For 3D visualization

# class CameraReaderNode(DTROS):

#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._bridge = CvBridge()

#         # Parameters for 3D reconstruction
#         self.prev_frame = None
#         self.prev_keypoints = None
#         self.prev_descriptors = None
#         self.frames = deque(maxlen=50)  # Store last 50 frames for processing
#         self.points_3d = []  # Store 3D points
#         self.colors = []  # Store colors for 3D points

#         # Feature detector (e.g., SIFT or ORB)
#         self.feature_detector = cv2.SIFT_create()  # or cv2.ORB_create()
#         self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # For SIFT

#         # Camera intrinsic parameters (you need to calibrate your Duckiebot camera)
#         self.K = np.array([[800, 0, 320],  # Example: focal length (fx, fy), center (cx, cy)
#                            [0, 800, 240],
#                            [0,   0,   1]], dtype=np.float32)

#         # Subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         curr_frame = self._bridge.compressed_imgmsg_to_cv2(msg)
#         curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

#         # Detect keypoints and descriptors
#         keypoints, descriptors = self.feature_detector.detectAndCompute(curr_frame_gray, None)

#         if self.prev_frame is not None and self.prev_descriptors is not None:
#             # Match features between previous and current frame
#             matches = self.matcher.match(self.prev_descriptors, descriptors)
#             matches = sorted(matches, key=lambda x: x.distance)

#             # Extract matched points
#             pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
#             pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

#             # Estimate essential matrix and recover pose
#             E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
#             _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)

#             # Triangulate points
#             pts1 = pts1[mask.ravel() == 1]
#             pts2 = pts2[mask.ravel() == 1]
#             points_4d = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), pts1.T, pts2.T)
#             points_3d = (points_4d[:3] / points_4d[3]).T

#             # Extract colors from the current frame
#             colors = [curr_frame[int(pt[1]), int(pt[0])] for pt in pts2]

#             # Store 3D points and colors
#             self.points_3d.extend(points_3d)
#             self.colors.extend(colors)

#             # Visualize in real-time (optional)
#             self.visualize_point_cloud()

#         # Update previous frame data
#         self.prev_frame = curr_frame.copy()
#         self.prev_keypoints = keypoints
#         self.prev_descriptors = descriptors

#         # Display current frame (optional)
#         cv2.imshow("Camera Feed", curr_frame)
#         cv2.waitKey(1)

#     def visualize_point_cloud(self):
#         if len(self.points_3d) > 0:
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(np.array(self.points_3d))
#             pcd.colors = o3d.utility.Vector3dVector(np.array(self.colors) / 255.0)  # Normalize to [0, 1]
#             o3d.visualization.draw_geometries([pcd])

# if __name__ == '__main__':
#     node = CameraReaderNode(node_name='camera_reader_node')
#     rospy.spin()














# #!/usr/bin/env python3

# import os
# import rospy
# import cv2
# import numpy as np
# import struct

# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
# from cv_bridge import CvBridge

# class CameraReaderNode(DTROS):
#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

#         # Camera intrinsics (These values depend on your camera)
#         self.focal_length = 300.0  # Adjust based on your camera
#         self.baseline = 0.06       # Adjust based on stereo setup (if applicable)
#         self.cx = 320              # Principal point x
#         self.cy = 240              # Principal point y

#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()

#         # Construct subscriber
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

#         # Point cloud publisher
#         self.pc_pub = rospy.Publisher(f"/{self._vehicle_name}/point_cloud", PointCloud2, queue_size=1)

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         image = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Create a fake stereo pair (shifted image)
#         shifted = np.roll(gray, shift=5, axis=1)

#         # Compute disparity map
#         stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
#         disparity = stereo.compute(gray, shifted).astype(np.float32)

#         # Convert disparity to depth (Avoid division by zero)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             depth_map = (self.focal_length * self.baseline) / (disparity + 1e-6)

#         # Convert depth map to 3D point cloud
#         points = self.depth_to_point_cloud(depth_map, image)

#         # Publish point cloud
#         self.publish_point_cloud(points)

#     def depth_to_point_cloud(self, depth_map, image):
#         """ Converts a depth map to a 3D point cloud. """
#         height, width = depth_map.shape
#         point_cloud = []

#         for v in range(height):
#             for u in range(width):
#                 depth = depth_map[v, u]
#                 if depth > 0 and depth < 10:  # Filter out invalid depth values
#                     x = (u - self.cx) * depth / self.focal_length
#                     y = (v - self.cy) * depth / self.focal_length
#                     z = depth

#                     # Get RGB values from image
#                     b, g, r = image[v, u]
#                     rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                    
#                     point_cloud.append((x, y, z, rgb))

#         return point_cloud

#     def publish_point_cloud(self, points):
#         """ Publishes a point cloud message to ROS. """
#         header = rospy.Header()
#         header.stamp = rospy.Time.now()
#         header.frame_id = "camera_link"

#         fields = [
#             PointField('x', 0, PointField.FLOAT32, 1),
#             PointField('y', 4, PointField.FLOAT32, 1),
#             PointField('z', 8, PointField.FLOAT32, 1),
#             PointField('rgb', 12, PointField.UINT32, 1),
#         ]

#         # Flatten point cloud data
#         cloud_data = np.array(points, dtype=np.float32).flatten().tobytes()

#         # Create PointCloud2 message
#         cloud_msg = PointCloud2(
#             header=header,
#             height=1,
#             width=len(points),
#             is_dense=False,
#             is_bigendian=False,
#             fields=fields,
#             point_step=16,
#             row_step=16 * len(points),
#             data=cloud_data
#         )

#         self.pc_pub.publish(cloud_msg)

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     rospy.spin()
