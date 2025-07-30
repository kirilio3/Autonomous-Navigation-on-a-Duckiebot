#!/usr/bin/env python3

import os
import math
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import ColorRGBA, Float64
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, LEDPattern
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
import signal
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge
from dt_apriltags import Detector

class DShapeNode(DTROS):
    def __init__(self, node_name):
        super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_params = None
        
        # Homography parameters
        self.src_points = np.float32([[50, 400], [620, 400], [360, 180], [250, 180]])
        self.dst_points = np.float32([[180, 500], [440, 500], [440, 0], [180, 0]])
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # AprilTag detector
        self.at_detector = Detector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Encoder variables
        self.last_left_ticks = None
        self.last_right_ticks = None
        self._left_distance_traveled = 0.0
        self._right_distance_traveled = 0.0
        
        # Parameters
        self.TICKS_PER_REV = 135
        self.WHEEL_RADIUS = 0.0318
        self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
        self.VELOCITY = 0.15
        self.OMEGA_SPEED = 3
        self.KP = 0.009
        self.KD = 0.01
        # self.KP = 0.0075
        # self.KD = 0.01
        self.TARGET_DISTANCE = 1.4
        
        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
        self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
        self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
        self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)

        # Subscribers
        self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
        self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)
        self.sub_camera_info = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/camera_info", CameraInfo, self.cb_camera_info)

        # Image processing variables
        self.bridge = CvBridge()
        self.latest_lane_center = None
        self.latest_tags = []
        
        # Control variables
        self.prev_error = 0.0
        self.prev_time = None
        
        signal.signal(signal.SIGINT, self.signal_handler)

    def draw_points(self, img, points, color=(0, 0, 255), thickness=5):
        img_with_points = img.copy()
        points = points.astype(int)
        for point in points:
            cv2.circle(img_with_points, tuple(point), thickness, color, -1)
        for i in range(len(points)):
            cv2.line(img_with_points, tuple(points[i]), tuple(points[(i + 1) % 4]), color, 2)
        return img_with_points
    
    def cb_camera_info(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)
        self.camera_params = (
            self.camera_matrix[0,0],  # fx
            self.camera_matrix[1,1],  # fy
            self.camera_matrix[0,2],  # cx
            self.camera_matrix[1,2]   # cy
        )
        rospy.loginfo("Camera parameters initialized")

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
            rospy.logwarn("Camera parameters not yet initialized")
            return
            
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg)
            undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
            image_with_points = self.draw_points(undistorted_image, self.src_points)
            gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
            # AprilTag detection
            self.latest_tags = self.at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=0.065
            )
            
            for tag in self.latest_tags:
                if tag.tag_id == 133:
                    distance = tag.pose_t[2][0]
                    rospy.loginfo(f"Detected AprilTag 133 at distance: {distance:.3f}m")
                    for i in range(4):
                        pt_a = tuple(tag.corners[i].astype(int))
                        pt_b = tuple(tag.corners[(i + 1) % 4].astype(int))
                        cv2.line(image_with_points, pt_a, pt_b, (0, 255, 0), 2)
                    cv2.putText(image_with_points, f"ID: {tag.tag_id}", 
                              tuple(tag.corners[0].astype(int)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Lane detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
            # Publish visualization
            original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            self.pub_original.publish(original_msg)
            self.pub_edges.publish(edges_msg)
            self.pub_birdseye.publish(birdseye_msg)
            
            # Update lane center
            self.latest_lane_center = self.detect_lanes_birdseye(undistorted_image)
            if self.latest_lane_center is not None:
                self.lane_center_pub.publish(Float64(self.latest_lane_center))
                
        except Exception as e:
            rospy.logerr(f"Error in camera callback: {str(e)}")

    def detect_lanes_birdseye(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            roi = birdseye[int(birdseye.shape[0]/2):, :]
            horizontal_sum = np.sum(roi, axis=0)
            if np.sum(horizontal_sum) > 1000:
                return np.average(np.arange(len(horizontal_sum)), weights=horizontal_sum)
            return None
        except Exception as e:
            rospy.logerr(f"Error in lane detection: {str(e)}")
            return None

    def cb_left_encoder(self, msg):
        current_ticks = msg.data
        if self.last_left_ticks is None:
            self.last_left_ticks = current_ticks
            return
        delta_ticks = current_ticks - self.last_left_ticks
        if delta_ticks > self.TICKS_PER_REV / 2:
            delta_ticks -= self.TICKS_PER_REV
        elif delta_ticks < -self.TICKS_PER_REV / 2:
            delta_ticks += self.TICKS_PER_REV
        self.last_left_ticks = current_ticks
        distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
        self._left_distance_traveled += distance

    def cb_right_encoder(self, msg):
        current_ticks = msg.data
        if self.last_right_ticks is None:
            self.last_right_ticks = current_ticks
            return
        delta_ticks = current_ticks - self.last_right_ticks
        if delta_ticks > self.TICKS_PER_REV / 2:
            delta_ticks -= self.TICKS_PER_REV
        elif delta_ticks < -self.TICKS_PER_REV / 2:
            delta_ticks += self.TICKS_PER_REV
        self.last_right_ticks = current_ticks
        distance = (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC
        self._right_distance_traveled += distance

    def set_led_color(self, color):
        pattern = LEDPattern()
        colors = {'GREEN': ColorRGBA(0, 1, 0, 1), 'RED': ColorRGBA(1, 0, 0, 1), 'CYAN': ColorRGBA(0, 1, 1, 1)}
        selected_color = colors.get(color, ColorRGBA(0.5, 0, 0.5, 1))
        pattern.color_list = [color] * 5
        pattern.rgb_vals = [selected_color] * 5
        pattern.color_mask = [1] * 5
        pattern.frequency = 1.0
        pattern.frequency_mask = [1] * 5
        self.led_pub.publish(pattern)

    def stop(self):
        msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd.publish(msg)

    def lane_follow(self):
        rospy.loginfo("Starting lane following with AprilTag detection...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(10)
        self.prev_time = rospy.get_time()
        phase = 1  # Track whether we're in the first or second lane-following phase
        
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            if avg_distance >= self.TARGET_DISTANCE:
                if phase == 1:
                    self.stop()
                    self.set_led_color('GREEN')
                    rospy.loginfo("First target distance reached! Moving back and turning right...")
                    # self.stop()
                    # rospy.sleep(2.0)
                    
                    # Move backward for 1 second
                    cmd = Twist2DStamped(v=-self.VELOCITY, omega=0.0)
                    self.pub_cmd.publish(cmd)
                    rospy.sleep(1.0)

                    # Turn right 45 degrees (approximate time-based turn)
                    cmd = Twist2DStamped(v=0.0, omega=self.OMEGA_SPEED)
                    self.pub_cmd.publish(cmd)
                    rospy.sleep(1.5)  # Adjust this value based on actual robot behavior
                    
                    # Move straight and to the left (forward with positive omega)
                    cmd = Twist2DStamped(v=self.VELOCITY*2, omega=self.OMEGA_SPEED)  # Positive omega for left turn
                    self.pub_cmd.publish(cmd)
                    rospy.sleep(1.4)  # Adjust duration to control how far it moves left

                    # Stop after maneuver
                    self.stop()
                    rospy.loginfo("Maneuver complete! Resetting distance and resuming lane following for 1.5m...")

                    # Reset distance traveled for the second phase
                    self._left_distance_traveled = 0.0
                    self._right_distance_traveled = 0.0
                    self.TARGET_DISTANCE = 1.5  # Set new target distance to 1.5 meters
                    self.set_led_color('CYAN')  # Indicate lane following is resuming
                    phase = 2  # Switch to second phase
                    continue
                elif phase == 2:
                    self.stop()
                    self._left_distance_traveled = 0.0
                    self._right_distance_traveled = 0.0
                    self.TARGET_DISTANCE = 1.0
                    self.set_led_color('GREEN')
                    rospy.loginfo("Second target distance (1.5m) reached! Stopping robot completely.")
                    break  # Exit the loop to stop completely
            
            # Check for AprilTag 133
            for tag in self.latest_tags:
                if tag.tag_id == 133:
                    distance = tag.pose_t[2][0]
                    if 0.12 <= distance <= 0.15:
                        rospy.loginfo(f"AprilTag 133 detected at {distance:.3f}m, stopping for 2 seconds")
                        self.stop()
                        self.set_led_color('RED')
                        rospy.sleep(2.0)
                        self.set_led_color('CYAN')
                        break

            # Lane following control
            if self.latest_lane_center is not None:
                image_center = 320
                error = image_center - self.latest_lane_center
                
                current_time = rospy.get_time()
                dt = current_time - self.prev_time if self.prev_time is not None and current_time > self.prev_time else 0.0
                error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                
                omega = (self.KP * error) + (self.KD * error_derivative)
                omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                
                cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                self.pub_cmd.publish(cmd)
                
                self.prev_error = error
                self.prev_time = current_time
            else:
                cmd = Twist2DStamped(v=self.VELOCITY/2, omega=0.0)
                self.pub_cmd.publish(cmd)
                rospy.logwarn("Lane center not detected, moving straight slowly")
            
            rate.sleep()

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.stop()
        super(DShapeNode, self).on_shutdown()

    def signal_handler(self, sig, frame):
        rospy.loginfo("Ctrl+C detected, shutting down...")
        self.on_shutdown()
        sys.exit(0)

    def run(self):
        rospy.sleep(0.5)  # Wait for subscribers/publishers to initialize
        self.lane_follow()

if __name__ == "__main__":
    node = DShapeNode(node_name="d_shape_node")
    node.run()
    rospy.loginfo("DShapeNode main finished.")










# #!/usr/bin/env python3

# import os
# import math
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from std_msgs.msg import ColorRGBA, Float64
# from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, LEDPattern
# from sensor_msgs.msg import CompressedImage, CameraInfo, Image
# import signal
# import sys
# import cv2
# import numpy as np
# from cv_bridge import CvBridge
# from dt_apriltags import Detector

# class DShapeNode(DTROS):
#     def __init__(self, node_name):
#         super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         self.vehicle_name = os.environ['VEHICLE_NAME']
        
#         # Camera calibration
#         self.camera_matrix = None
#         self.distortion_coeffs = None
        
#         # Homography for bird's-eye view (lane following)
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
        
#         # Encoder variables
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
        
#         # Parameters
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
#         self.BASELINE = 0.077
#         self.VELOCITY = 0.15
#         self.OMEGA_SPEED = 5
#         self.angular_vel = 2.6
        
#         # Control parameters
#         self.KP = 0.0075
#         self.KD = 0.01
#         self.TARGET_DISTANCE = 1.35
        
#         # AprilTag detector
#         self.at_detector = Detector(
#             families='tag36h11',
#             nthreads=1,
#             quad_decimate=1.0,
#             quad_sigma=0.0,
#             refine_edges=1,
#             decode_sharpening=0.25,
#             debug=0
#         )
        
#         # Publishers
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
#         self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
#         self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
#         self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)

#         # Subscribers
#         self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
#         self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
#         self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
#         self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"
        
#         self.sub_left_enc = rospy.Subscriber(self.left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(self.right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)
#         self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info)

#         # Image processing
#         self.bridge = CvBridge()
#         self.prev_error = 0.0
#         self.prev_time = None

#         signal.signal(signal.SIGINT, self.signal_handler)

#     def draw_points(self, img, points, color=(0, 0, 255), thickness=5):
#         img_with_points = img.copy()
#         points = points.astype(int)
#         for point in points:
#             cv2.circle(img_with_points, tuple(point), thickness, color, -1)
#         for i in range(len(points)):
#             cv2.line(img_with_points, tuple(points[i]), tuple(points[(i + 1) % 4]), color, 2)
#         return img_with_points
    
#     def cb_camera_info(self, msg):
#         self.camera_matrix = np.array(msg.K).reshape(3, 3)
#         self.distortion_coeffs = np.array(msg.D)

#     def detect_apriltags(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         tags = self.at_detector.detect(
#             gray,
#             estimate_tag_pose=True,
#             camera_params=[self.camera_matrix[0,0], self.camera_matrix[1,1], 
#                          self.camera_matrix[0,2], self.camera_matrix[1,2]],
#             tag_size=0.065  # Adjust based on your AprilTag size in meters
#         )
#         return tags

#     def cb_camera(self, msg):
#         if self.camera_matrix is None or self.distortion_coeffs is None:
#             return
            
#         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)

#         # AprilTag detection
#         tags = self.detect_apriltags(undistorted_image)
#         for tag in tags:
#             if tag.tag_id == 133:
#                 # Distance is in meters (from pose_t), convert to cm
#                 distance_cm = tag.pose_t[2][0] * 100
#                 if 8 <= distance_cm <= 12:  # Allow some tolerance around 10cm
#                     self.stop()
#                     self.set_led_color('RED')
#                     rospy.loginfo(f"AprilTag 133 detected at {distance_cm:.1f}cm, stopping for 2 seconds")
#                     rospy.sleep(2)
#                     self.set_led_color('CYAN')

#         # Lane following processing
#         image_with_points = self.draw_points(undistorted_image, self.src_points)
#         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 50, 150)
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
#         # Publish visualization images
#         try:
#             original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#             edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#             birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#             self.pub_original.publish(original_msg)
#             self.pub_edges.publish(edges_msg)
#             self.pub_birdseye.publish(birdseye_msg)
#         except Exception as e:
#             rospy.logerr(f"Error publishing visualization images: {str(e)}")
        
#         lane_center = self.detect_lanes_birdseye(undistorted_image)
#         if lane_center is not None:
#             self.lane_center_pub.publish(Float64(lane_center))

#     def detect_lanes_birdseye(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 50, 150)
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
#         roi = birdseye[int(birdseye.shape[0]/2):, :]
#         horizontal_sum = np.sum(roi, axis=0)
#         if np.sum(horizontal_sum) > 1000:
#             lane_center = np.average(np.arange(len(horizontal_sum)), weights=horizontal_sum)
#             return lane_center
#         return None

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

#     def set_led_color(self, color):
#         pattern = LEDPattern()
#         colors = {
#             'GREEN': ColorRGBA(0, 1, 0, 1),
#             'RED': ColorRGBA(1, 0, 0, 1),
#             'CYAN': ColorRGBA(0, 1, 1, 1)
#         }
#         selected_color = colors.get(color, ColorRGBA(0.5, 0, 0.5, 1))
#         pattern.color_list = [color] * 5
#         pattern.rgb_vals = [selected_color] * 5
#         pattern.color_mask = [1] * 5
#         pattern.frequency = 1.0
#         pattern.frequency_mask = [1] * 5
#         self.led_pub.publish(pattern)

#     def lane_follow(self):
#         rospy.loginfo("Starting lane following for 1.35 meters with PD control and AprilTag detection...")
#         self.set_led_color('CYAN')
#         rate = rospy.Rate(10)
#         self.prev_time = rospy.get_time()
        
#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
#             if avg_distance >= self.TARGET_DISTANCE:
#                 self.stop()
#                 self.set_led_color('GREEN')
#                 rospy.loginfo("Target distance reached!")
#                 break
                
#             try:
#                 lane_msg = rospy.wait_for_message(f"/{self.vehicle_name}/lane_center", Float64, timeout=1.0)
#                 lane_center = lane_msg.data
#                 image_center = 320
#                 error = image_center - lane_center
#                 current_time = rospy.get_time()
#                 if self.prev_time is not None:
#                     dt = current_time - self.prev_time
#                     error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
#                 else:
#                     error_derivative = 0.0
                
#                 omega = (self.KP * error) + (self.KD * error_derivative)
#                 omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
#                 cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#                 self.pub_cmd.publish(cmd)
                
#                 self.prev_error = error
#                 self.prev_time = current_time
                
#             except rospy.ROSException:
#                 cmd = Twist2DStamped(v=self.VELOCITY/2, omega=0.0)
#                 self.pub_cmd.publish(cmd)
#                 rospy.logwarn("Lane center not detected, moving straight slowly")
                
#             rate.sleep()

#     def stop(self):
#         msg = Twist2DStamped(v=0.0, omega=0.0)
#         self.pub_cmd.publish(msg)

#     def on_shutdown(self):
#         rospy.loginfo("Shutting down node...")
#         self.stop()
#         super(DShapeNode, self).on_shutdown()

#     def signal_handler(self, sig, frame):
#         rospy.loginfo("Ctrl+C detected, shutting down...")
#         self.on_shutdown()
#         sys.exit(0)

#     def run(self):
#         rospy.sleep(0.5)
#         self.lane_follow()

# if __name__ == "__main__":
#     node = DShapeNode(node_name="d_shape_node")
#     node.run()
#     rospy.loginfo("DShapeNode main finished.")





# #!/usr/bin/env python3

# import os
# import math
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from std_msgs.msg import ColorRGBA, Float64
# from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, LEDPattern
# from sensor_msgs.msg import CompressedImage, CameraInfo, Image
# import signal
# import sys
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class DShapeNode(DTROS):
#     def __init__(self, node_name):
#         super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         self.vehicle_name = os.environ['VEHICLE_NAME']
        
#         # Camera calibration
#         self.camera_matrix = None
#         self.distortion_coeffs = None
        
#         # Homography for bird's-eye view (lane following)
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
        
#         # Encoder variables
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
        
#         # Parameters
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
#         self.BASELINE = 0.077
#         self.VELOCITY = 0.15
#         self.OMEGA_SPEED = 5
#         self.angular_vel = 2.6
        
#         # Control parameters
#         self.KP = 0.0075
#         self.KD = 0.01
#         self.TARGET_DISTANCE = 1.35
        
#         # Publishers
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
#         self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
#         self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
#         self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
#         # AR output topic
#         self.ar_output_topic = f"/{self.vehicle_name}/ar_planar_node/image/compressed"
#         self.pub_ar = rospy.Publisher(self.ar_output_topic, CompressedImage, queue_size=1)

#         # Subscribers
#         self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
#         self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
#         self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
#         self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"
        
#         self.sub_left_enc = rospy.Subscriber(self.left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(self.right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)
#         self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info)

#         # Image processing
#         self.bridge = CvBridge()
#         self.prev_error = 0.0
#         self.prev_time = None
        
#         # AR-specific initialization
#         self.detector = cv2.ORB.create(nfeatures=500)
#         package_path = os.path.dirname(os.path.realpath(__file__))
#         image_path = os.path.join(package_path, "images/apriltag.png")
#         self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         if self.overlay_img is None:
#             self.logerr("Could not load overlay image")
#             rospy.signal_shutdown("Overlay image not found")
#             return
        
#         # Convert overlay image to 3-channel BGR if needed
#         if len(self.overlay_img.shape) == 2:  # Grayscale
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGR)
#         elif self.overlay_img.shape[2] == 4:  # RGBA
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGRA2BGR)
#         self.overlay_img = cv2.resize(self.overlay_img, (100, 100))  # Adjust size as needed
        
#         signal.signal(signal.SIGINT, self.signal_handler)

#     def draw_points(self, img, points, color=(0, 0, 255), thickness=5):
#         img_with_points = img.copy()
#         points = points.astype(int)
#         for point in points:
#             cv2.circle(img_with_points, tuple(point), thickness, color, -1)
#         for i in range(len(points)):
#             cv2.line(img_with_points, tuple(points[i]), tuple(points[(i + 1) % 4]), color, 2)
#         return img_with_points
    
#     def cb_camera_info(self, msg):
#         self.camera_matrix = np.array(msg.K).reshape(3, 3)
#         self.distortion_coeffs = np.array(msg.D)

#     def cb_camera(self, msg):
#         if self.camera_matrix is None or self.distortion_coeffs is None:
#             return
            
#         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)

#         # Lane following processing
#         image_with_points = self.draw_points(undistorted_image, self.src_points)
#         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 50, 150)
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
#         # Publish visualization images
#         try:
#             original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#             edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#             birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#             self.pub_original.publish(original_msg)
#             self.pub_edges.publish(edges_msg)
#             self.pub_birdseye.publish(birdseye_msg)
#         except Exception as e:
#             rospy.logerr(f"Error publishing visualization images: {str(e)}")
        
#         lane_center = self.detect_lanes_birdseye(undistorted_image)
#         if lane_center is not None:
#             self.lane_center_pub.publish(Float64(lane_center))

#         # AR processing
#         frame_with_ar = self.process_ar(undistorted_image)
#         ar_output_msg = self.bridge.cv2_to_compressed_imgmsg(frame_with_ar)
#         ar_output_msg.header = msg.header
#         self.pub_ar.publish(ar_output_msg)

#     def detect_lanes_birdseye(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 50, 150)
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
#         roi = birdseye[int(birdseye.shape[0]/2):, :]
#         horizontal_sum = np.sum(roi, axis=0)
#         if np.sum(horizontal_sum) > 1000:
#             lane_center = np.average(np.arange(len(horizontal_sum)), weights=horizontal_sum)
#             return lane_center
#         return None

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

#     def set_led_color(self, color):
#         pattern = LEDPattern()
#         colors = {
#             'GREEN': ColorRGBA(0, 1, 0, 1),
#             'RED': ColorRGBA(1, 0, 0, 1),
#             'CYAN': ColorRGBA(0, 1, 1, 1)
#         }
#         selected_color = colors.get(color, ColorRGBA(0.5, 0, 0.5, 1))
#         pattern.color_list = [color] * 5
#         pattern.rgb_vals = [selected_color] * 5
#         pattern.color_mask = [1] * 5
#         pattern.frequency = 1.0
#         pattern.frequency_mask = [1] * 5
#         self.led_pub.publish(pattern)

#     def stop(self):
#         msg = Twist2DStamped(v=0.0, omega=0.0)
#         self.pub_cmd.publish(msg)

#     # AR-specific methods
#     def find_planar_regions(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         keypoints, descriptors = self.detector.detectAndCompute(gray, None)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#         edged = cv2.Canny(blurred, 50, 150)
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         largest_region = None
#         max_area = 0
#         for cnt in contours:
#             perimeter = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
#             if len(approx) == 4:
#                 area = cv2.contourArea(approx)
#                 if area > 1000 and cv2.isContourConvex(approx) and area > max_area:
#                     max_area = area
#                     largest_region = self.order_points(approx.reshape(4, 2))
#         return largest_region

#     def order_points(self, pts):
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect

#     def compute_homography(self, src_pts, dst_pts):
#         H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         if H is None:
#             self.logerr("Homography computation failed.")
#             return None
#         return H.astype(np.float32)

#     def warp_image(self, image, H, dst_shape):
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))
#         return warped

#     def process_ar(self, frame):
#         largest_region = self.find_planar_regions(frame)
#         if largest_region is not None:
#             h, w = self.overlay_img.shape[:2]
#             src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
#             H = self.compute_homography(src_pts, largest_region.astype(np.float32))
#             if H is not None:
#                 warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
#                 mask = np.zeros_like(frame, dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, np.int32(largest_region), (255, 255, 255))
#                 mask = cv2.bitwise_not(mask)
#                 frame_masked = cv2.bitwise_and(frame, mask)
#                 frame = cv2.bitwise_or(frame_masked, warped_overlay)
#         return frame

#     def lane_follow(self):
#         rospy.loginfo("Starting lane following for 1.35 meters with PD control and AR overlay...")
#         self.set_led_color('CYAN')
#         rate = rospy.Rate(10)
#         self.prev_time = rospy.get_time()
        
#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
#             if avg_distance >= self.TARGET_DISTANCE:
#                 self.stop()
#                 self.set_led_color('GREEN')
#                 rospy.loginfo("Target distance reached!")
#                 break
                
#             try:
#                 lane_msg = rospy.wait_for_message(f"/{self.vehicle_name}/lane_center", Float64, timeout=1.0)
#                 lane_center = lane_msg.data
#                 image_center = 320
#                 error = image_center - lane_center
#                 current_time = rospy.get_time()
#                 if self.prev_time is not None:
#                     dt = current_time - self.prev_time
#                     error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
#                 else:
#                     error_derivative = 0.0
                
#                 omega = (self.KP * error) + (self.KD * error_derivative)
#                 omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
#                 cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#                 self.pub_cmd.publish(cmd)
                
#                 self.prev_error = error
#                 self.prev_time = current_time
#             except rospy.ROSException:
#                 cmd = Twist2DStamped(v=self.VELOCITY/2, omega=0.0)
#                 self.pub_cmd.publish(cmd)
#                 rospy.logwarn("Lane center not detected, moving straight slowly")
                
#             rate.sleep()

#     def on_shutdown(self):
#         rospy.loginfo("Shutting down node...")
#         self.stop()
#         super(DShapeNode, self).on_shutdown()

#     def signal_handler(self, sig, frame):
#         rospy.loginfo("Ctrl+C detected, shutting down...")
#         self.on_shutdown()
#         sys.exit(0)

#     def run(self):
#         rospy.sleep(0.5)
#         self.lane_follow()

# if __name__ == "__main__":
#     node = DShapeNode(node_name="d_shape_node")
#     node.run()
#     rospy.loginfo("DShapeNode main finished.")