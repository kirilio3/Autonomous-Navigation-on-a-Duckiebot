import cv2
print(cv2.__version__)
print(hasattr(cv2, 'xfeatures2d'))  # Should return True
print(cv2.xfeatures2d.SIFT_create())  # Should not raise an error




# #!/usr/bin/env python3

# import os
# import math
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
# from sensor_msgs.msg import CompressedImage, Image
# from std_msgs.msg import Float64
# import cv2
# import signal
# import numpy as np
# from cv_bridge import CvBridge
# import sys

# class Lane_Following(DTROS):
#     def __init__(self, node_name='lane_following_node', vehicle_name=None):
#         super(Lane_Following, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         self.vehicle_name = vehicle_name or os.environ.get('VEHICLE_NAME', 'duckiebot')

#         # Bridge for image conversion
#         self.bridge = CvBridge()

#         # Encoder variables
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0

#         # Parameters
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
#         self.VELOCITY = 0.2
#         self.OMEGA_SPEED = 2.5

#         # Control parameters
#         self.KP = 0.015
#         self.KI = 0.0001
#         self.KD = 0.01
#         self.TARGET_DISTANCE = 20  # meters

#         # PID variables
#         self.prev_error = 0.0
#         self.integral = 0.0
#         self.prev_time = None

#         # Publishers
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.left_edge_pub = rospy.Publisher(f"/{self.vehicle_name}/left_edge", Float64, queue_size=1)
#         self.right_edge_pub = rospy.Publisher(f"/{self.vehicle_name}/right_edge", Float64, queue_size=1)
#         # Visualization publishers
#         self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)

#         # Subscribers
#         self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)

#         # Homography setup (tune these points based on your camera and environment)
#         self.src_points = np.float32([
#             [50, 400],  # bottom-left
#             [620, 400], # bottom-right
#             [360, 180], # top-right
#             [250, 180]  # top-left
#         ])
#         self.dst_points = np.float32([
#             [180, 500], # bottom-left
#             [440, 500], # bottom-right
#             [440, 0],   # top-right
#             [180, 0]    # top-left
#         ])
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
#         signal.signal(signal.SIGINT, self.signal_handler)

#     def draw_points(self, img, points, color=(0, 0, 255), thickness=5):
#         """Draw points and connect them with lines on the image"""
#         img_with_points = img.copy()
#         points = points.astype(int)
        
#         # Draw points
#         for point in points:
#             cv2.circle(img_with_points, tuple(point), thickness, color, -1)
        
#         # Draw connecting lines
#         for i in range(len(points)):
#             cv2.line(img_with_points, tuple(points[i]), tuple(points[(i + 1) % 4]), color, 2)
        
#         return img_with_points

#     def cb_camera(self, msg):
#         # Convert compressed image to OpenCV format
#         image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Process image with homography and edge detection
#         left_edge, right_edge, original_with_points, edges, birdseye = self.process_image(image)

#         # Publish edge positions
#         if left_edge is not None:
#             self.left_edge_pub.publish(Float64(left_edge))
#         if right_edge is not None:
#             self.right_edge_pub.publish(Float64(right_edge))

#         # Publish visualization images
#         try:
#             original_msg = self.bridge.cv2_to_imgmsg(original_with_points, encoding="bgr8")
#             edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#             birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            
#             self.pub_original.publish(original_msg)
#             self.pub_edges.publish(edges_msg)
#             self.pub_birdseye.publish(birdseye_msg)
#         except Exception as e:
#             rospy.logerr(f"Error publishing visualization images: {str(e)}")

#     def process_image(self, img):
#         # Draw source points on original image
#         original_with_points = self.draw_points(img, self.src_points)
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Edge detection using Canny
#         edges = cv2.Canny(blurred, 50, 150)
        
#         # Apply perspective transform to get bird's-eye view
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))

#         # Detect lane edges in bird's-eye view
#         left_edge, right_edge = self.detect_lane_edges(birdseye)

#         # Draw lane edges on the birdseye image
#         if left_edge is not None:
#             cv2.line(birdseye, (int(left_edge), 0), (int(left_edge), birdseye.shape[0]), 255, 2)
#         if right_edge is not None:
#             cv2.line(birdseye, (int(right_edge), 0), (int(right_edge), birdseye.shape[0]), 255, 2)
        
#         # Draw lane center if both edges are detected
#         if left_edge is not None and right_edge is not None:
#             lane_center = (left_edge + right_edge) / 2
#             cv2.line(birdseye, (int(lane_center), 0), (int(lane_center), birdseye.shape[0]), 128, 2)

#         return left_edge, right_edge, original_with_points, edges, birdseye

#     def detect_lane_edges(self, edges):
#         # Crop the lower part of the image for lane detection
#         h, w = edges.shape
#         roi = edges[h//2:, :]  # Focus on the lower half of the bird's-eye view

#         # Use histogram to find lane edges
#         histogram = np.sum(roi, axis=0)
#         midpoint = w // 2

#         # Find left edge
#         left_region = histogram[:midpoint]
#         left_edge = np.argmax(left_region) if np.max(left_region) > 500 else None  # Threshold to ensure edge presence

#         # Find right edge
#         right_region = histogram[midpoint:]
#         right_edge_idx = np.argmax(right_region) if np.max(right_region) > 500 else None
#         right_edge = midpoint + right_edge_idx if right_edge_idx is not None else None

#         return left_edge, right_edge

#     def cb_left_encoder(self, msg):
#         current_ticks = msg.data
#         if self.last_left_ticks is None:
#             self.last_left_ticks = current_ticks
#             return
#         delta_ticks = current_ticks - self.last_left_ticks
#         if delta_ticks > self.TICKS_PER_REV / 2:
#             delta_ticks -= self.TICKS_PER_REV
#         elif delta_ticks < -self.TICKS_PER_REV / 2:
#             delta_ticks += self.TICKS_PER_REV
#         self.last_left_ticks = current_ticks
#         distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
#         self._left_distance_traveled += distance

#     def cb_right_encoder(self, msg):
#         current_ticks = msg.data
#         if self.last_right_ticks is None:
#             self.last_right_ticks = current_ticks
#             return
#         delta_ticks = current_ticks - self.last_right_ticks
#         if delta_ticks > self.TICKS_PER_REV / 2:
#             delta_ticks -= self.TICKS_PER_REV
#         elif delta_ticks < -self.TICKS_PER_REV / 2:
#             delta_ticks += self.TICKS_PER_REV
#         self.last_right_ticks = current_ticks
#         distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
#         self._right_distance_traveled += distance

#     def stop(self):
#         msg = Twist2DStamped(v=0.0, omega=0.0)
#         self.pub_cmd.publish(msg)

#     def pid_control(self, error):
#         current_time = rospy.get_time()
#         dt = current_time - self.prev_time if self.prev_time is not None else 0.001
#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
#         omega = (self.KP * error) + (self.KI * self.integral) + (self.KD * derivative)
#         omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
#         cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#         self.pub_cmd.publish(cmd)
#         self.prev_error = error
#         self.prev_time = current_time

#     def lane_follow(self, distance, rate=20):
#         while (True):
#             pass
#         # rospy.loginfo(f"Starting lane following for {distance} meters with PID control...")
#         # rate = rospy.Rate(rate)
#         # self.prev_time = rospy.get_time()

#         # while not rospy.is_shutdown():
#         #     avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
#         #     if avg_distance >= distance:
#         #         self.stop()
#         #         rospy.loginfo("Target distance reached!")
#         #         break

#         #     try:
#         #         left_msg = rospy.wait_for_message(f"/{self.vehicle_name}/left_edge", Float64, timeout=1.0)
#         #         right_msg = rospy.wait_for_message(f"/{self.vehicle_name}/right_edge", Float64, timeout=1.0)
#         #     except rospy.ROSException:
#         #         rospy.logwarn("Failed to get edge messages, moving straight slowly")
#         #         cmd = Twist2DStamped(v=self.VELOCITY / 1.5, omega=0)
#         #         self.pub_cmd.publish(cmd)
#         #         rate.sleep()
#         #         continue

#         #     if left_msg.data is not None and right_msg.data is not None:
#         #         lane_center = (left_msg.data + right_msg.data) / 2
#         #         image_center = 320  # Center of 640px wide bird's-eye view
#         #         error = image_center - lane_center
#         #         self.pid_control(error)
#         #     else:
#         #         rospy.logwarn("Lane edges not detected, moving straight slowly")
#         #         cmd = Twist2DStamped(v=self.VELOCITY / 1.5, omega=0)
#         #         self.pub_cmd.publish(cmd)

#         #     rate.sleep()

#     def on_shutdown(self):
#         rospy.loginfo("Shutting down node...")
#         self.stop()
#         super(Lane_Following, self).on_shutdown()

#     def signal_handler(self, sig, frame):
#         rospy.loginfo("Ctrl+C detected, shutting down...")
#         self.on_shutdown()
#         sys.exit(0)

#     def run(self):
#         rospy.sleep(0.5)
#         self.lane_follow(distance=2)

# if __name__ == "__main__":
#     node = Lane_Following(node_name="lane_following")
#     node.run()
#     rospy.loginfo("Lane Following main finished.")




# #!/usr/bin/env python3

# import os
# import math
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
# from sensor_msgs.msg import CompressedImage
# from std_msgs.msg import Float64
# import cv2
# import signal
# import numpy as np
# from cv_bridge import CvBridge
# import sys

# class Lane_Following(DTROS):
#     def __init__(self, node_name='lane_following_node', vehicle_name=None):
#         super(Lane_Following, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         self.vehicle_name = vehicle_name or os.environ.get('VEHICLE_NAME', 'duckiebot')

#         # Bridge for image conversion
#         self.bridge = CvBridge()

#         # Encoder variables
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0

#         # Parameters
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
#         self.VELOCITY = 0.2
#         self.OMEGA_SPEED = 2.5

#         # Control parameters
#         self.KP = 0.015
#         self.KI = 0.0001
#         self.KD = 0.01
#         self.TARGET_DISTANCE = 20  # meters

#         # PID variables
#         self.prev_error = 0.0
#         self.integral = 0.0
#         self.prev_time = None

#         # Publishers
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.left_edge_pub = rospy.Publisher(f"/{self.vehicle_name}/left_edge", Float64, queue_size=1)
#         self.right_edge_pub = rospy.Publisher(f"/{self.vehicle_name}/right_edge", Float64, queue_size=1)
#         self.image_pub = rospy.Publisher(f"/{self.vehicle_name}/birdseye_image/compressed", CompressedImage, queue_size=10)

#         # Subscribers
#         self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)

#         # Homography setup (tune these points based on your camera and environment)
#         self.src_points = np.float32([
#             [50, 400],  # bottom-left
#             [620, 400], # bottom-right
#             [360, 180], # top-right
#             [250, 180]  # top-left
#         ])
#         self.dst_points = np.float32([
#             [180, 500], # bottom-left
#             [440, 500], # bottom-right
#             [440, 0],   # top-right
#             [180, 0]    # top-left
#         ])
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
#         signal.signal(signal.SIGINT, self.signal_handler)

#     def cb_camera(self, msg):
#         # Convert compressed image to OpenCV format
#         image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Process image with homography and edge detection
#         left_edge, right_edge, birdseye = self.process_image(image)

#         # Publish edge positions
#         if left_edge is not None:
#             self.left_edge_pub.publish(Float64(left_edge))
#         if right_edge is not None:
#             self.right_edge_pub.publish(Float64(right_edge))

#         # Publish bird's-eye view image for visualization
#         birdseye_msg = self.bridge.cv2_to_compressed_imgmsg(birdseye)
#         self.image_pub.publish(birdseye_msg)

#     def process_image(self, img):
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur to reduce noise
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Edge detection using Canny
#         edges = cv2.Canny(blurred, 50, 150)
        
#         # Apply perspective transform to get bird's-eye view
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))

#         # Detect lane edges in bird's-eye view
#         left_edge, right_edge = self.detect_lane_edges(birdseye)

#         # Draw lane edges on the birdseye image
#         if left_edge is not None:
#             cv2.line(birdseye, (int(left_edge), 0), (int(left_edge), birdseye.shape[0]), 255, 2)
#         if right_edge is not None:
#             cv2.line(birdseye, (int(right_edge), 0), (int(right_edge), birdseye.shape[0]), 255, 2)
        
#         # Draw lane center if both edges are detected
#         if left_edge is not None and right_edge is not None:
#             lane_center = (left_edge + right_edge) / 2
#             cv2.line(birdseye, (int(lane_center), 0), (int(lane_center), birdseye.shape[0]), 128, 2)

#         return left_edge, right_edge, birdseye

#     def detect_lane_edges(self, edges):
#         # Crop the lower part of the image for lane detection
#         h, w = edges.shape
#         roi = edges[h//2:, :]  # Focus on the lower half of the bird's-eye view

#         # Use histogram to find lane edges
#         histogram = np.sum(roi, axis=0)
#         midpoint = w // 2

#         # Find left edge
#         left_region = histogram[:midpoint]
#         left_edge = np.argmax(left_region) if np.max(left_region) > 500 else None  # Threshold to ensure edge presence

#         # Find right edge
#         right_region = histogram[midpoint:]
#         right_edge_idx = np.argmax(right_region) if np.max(right_region) > 500 else None
#         right_edge = midpoint + right_edge_idx if right_edge_idx is not None else None

#         return left_edge, right_edge

#     def cb_left_encoder(self, msg):
#         current_ticks = msg.data
#         if self.last_left_ticks is None:
#             self.last_left_ticks = current_ticks
#             return
#         delta_ticks = current_ticks - self.last_left_ticks
#         if delta_ticks > self.TICKS_PER_REV / 2:
#             delta_ticks -= self.TICKS_PER_REV
#         elif delta_ticks < -self.TICKS_PER_REV / 2:
#             delta_ticks += self.TICKS_PER_REV
#         self.last_left_ticks = current_ticks
#         distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
#         self._left_distance_traveled += distance

#     def cb_right_encoder(self, msg):
#         current_ticks = msg.data
#         if self.last_right_ticks is None:
#             self.last_right_ticks = current_ticks
#             return
#         delta_ticks = current_ticks - self.last_right_ticks
#         if delta_ticks > self.TICKS_PER_REV / 2:
#             delta_ticks -= self.TICKS_PER_REV
#         elif delta_ticks < -self.TICKS_PER_REV / 2:
#             delta_ticks += self.TICKS_PER_REV
#         self.last_right_ticks = current_ticks
#         distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
#         self._right_distance_traveled += distance

#     def stop(self):
#         msg = Twist2DStamped(v=0.0, omega=0.0)
#         self.pub_cmd.publish(msg)

#     def pid_control(self, error):
#         current_time = rospy.get_time()
#         dt = current_time - self.prev_time if self.prev_time is not None else 0.001
#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
#         omega = (self.KP * error) + (self.KI * self.integral) + (self.KD * derivative)
#         omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
#         cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#         self.pub_cmd.publish(cmd)
#         self.prev_error = error
#         self.prev_time = current_time

#     def lane_follow(self, distance, rate=20):
#         rospy.loginfo(f"Starting lane following for {distance} meters with PID control...")
#         rate = rospy.Rate(rate)
#         self.prev_time = rospy.get_time()

#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
#             if avg_distance >= distance:
#                 self.stop()
#                 rospy.loginfo("Target distance reached!")
#                 break

#             try:
#                 left_msg = rospy.wait_for_message(f"/{self.vehicle_name}/left_edge", Float64, timeout=1.0)
#                 right_msg = rospy.wait_for_message(f"/{self.vehicle_name}/right_edge", Float64, timeout=1.0)
#             except rospy.ROSException:
#                 rospy.logwarn("Failed to get edge messages, moving straight slowly")
#                 cmd = Twist2DStamped(v=self.VELOCITY / 1.5, omega=0)
#                 self.pub_cmd.publish(cmd)
#                 rate.sleep()
#                 continue

#             if left_msg.data is not None and right_msg.data is not None:
#                 lane_center = (left_msg.data + right_msg.data) / 2
#                 image_center = 320  # Center of 640px wide bird's-eye view
#                 error = image_center - lane_center
#                 self.pid_control(error)
#             else:
#                 rospy.logwarn("Lane edges not detected, moving straight slowly")
#                 cmd = Twist2DStamped(v=self.VELOCITY / 1.5, omega=0)
#                 self.pub_cmd.publish(cmd)

#             rate.sleep()

#     def on_shutdown(self):
#         rospy.loginfo("Shutting down node...")
#         self.stop()
#         super(Lane_Following, self).on_shutdown()

#     def signal_handler(self, sig, frame):
#         rospy.loginfo("Ctrl+C detected, shutting down...")
#         self.on_shutdown()
#         sys.exit(0)

#     def run(self):
#         rospy.sleep(0.5)
#         self.lane_follow(distance=20)

# if __name__ == "__main__":
#     node = Lane_Following(node_name="lane_following")
#     node.run()
#     rospy.loginfo("Lane Following main finished.")

# if __name__ == '__main__':
#     node = Lane_Following()
#     try:
#         node.lane_follow(distance=20)
#         # rospy.spin()
#     except KeyboardInterrupt:
#         node.stop()
