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

class DShapeNode(DTROS):
    def __init__(self, node_name):
        super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Existing initialization code...
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Homography parameters
        self.src_points = np.float32([
            [50, 400], [620, 400], [360, 180], [250, 180]
        ])
        self.dst_points = np.float32([
            [180, 500], [440, 500], [440, 0], [180, 0]
        ])
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
        # Encoder variables
        self.last_left_ticks = None
        self.last_right_ticks = None
        self._left_distance_traveled = 0.0
        self._right_distance_traveled = 0.0
        
        # Parameters
        self.TICKS_PER_REV = 135
        self.WHEEL_RADIUS = 0.0318
        self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
        self.BASELINE = 0.077
        self.VELOCITY = 0.15
        self.OMEGA_SPEED = 2.5
        self.angular_vel = 2.6
        
        # Control parameters
        self.KP = 0.0075
        self.KD = 0.01
        self.TARGET_DISTANCE = 1.35
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Feature parameters for good features to track
        self.feature_params = dict(maxCorners=10,
                                 qualityLevel=0.9,
                                 minDistance=1,
                                 blockSize=32)
        
        self.prev_gray = None
        self.prev_points = None
        self.obstacle_detected = False
        self.MIN_OBSTACLE_DISTANCE = 0.1  # meters
        
        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self.led_pub = rospy.Publisher(f"/{self.vehicle_name}/led_emitter_node/led_pattern", LEDPattern, queue_size=1)
        self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
        
        # Visualization publishers
        self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
        self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
        self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
        
        self.pub_flow = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/optical_flow", Image, queue_size=1)

        # Subscribers
        self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
        self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)
        self.sub_camera_info = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/camera_info", CameraInfo, self.cb_camera_info)

        self.bridge = CvBridge()
        self.prev_error = 0.0
        self.prev_time = None
        
        signal.signal(signal.SIGINT, self.signal_handler)

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
    
    def cb_camera_info(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.D)

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None:
            return
            
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        
        # Optical flow processing
        gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        flow_image = self.process_optical_flow(undistorted_image, gray)
        
        # Lane following processing
        image_with_points = self.draw_points(undistorted_image, self.src_points)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
        # Publish visualization images
        try:
            self.pub_original.publish(self.bridge.cv2_to_imgmsg(image_with_points, "bgr8"))
            self.pub_edges.publish(self.bridge.cv2_to_imgmsg(edges, "mono8"))
            self.pub_birdseye.publish(self.bridge.cv2_to_imgmsg(birdseye, "mono8"))
            self.pub_flow.publish(self.bridge.cv2_to_imgmsg(flow_image, "bgr8"))
        except Exception as e:
            rospy.logerr(f"Error publishing visualization images: {str(e)}")
        
        lane_center = self.detect_lanes_birdseye(undistorted_image)
        if lane_center is not None:
            self.lane_center_pub.publish(Float64(lane_center))

    def process_optical_flow(self, image, gray):
        flow_image = image.copy()
        
        # Initialize previous frame and points if not already done
        if self.prev_gray is None or self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            return flow_image
        
        # Calculate optical flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        # Check if optical flow calculation failed
        if next_points is None or status is None or len(next_points) == 0:
            rospy.logwarn("Optical flow calculation failed, reinitializing feature points")
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            self.obstacle_detected = False
            return flow_image
        
        # Select good points
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        # Calculate flow magnitude and check for obstacles
        if len(good_new) > 0 and len(good_old) > 0:
            flow_vectors = good_new - good_old
            magnitudes = np.sqrt(np.sum(flow_vectors**2, axis=1))
            
            # Simple distance estimation
            avg_magnitude = np.mean(magnitudes)
            estimated_distance = 1.0 / (avg_magnitude + 1e-6)  # Avoid division by zero
            
            # Check for obstacles
            if estimated_distance < self.MIN_OBSTACLE_DISTANCE and avg_magnitude > 5.0:
                self.obstacle_detected = True
                rospy.logwarn(f"Obstacle detected at estimated distance: {estimated_distance:.3f}m")
            else:
                self.obstacle_detected = False
                
            # Draw flow vectors
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                flow_image = cv2.line(flow_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                flow_image = cv2.circle(flow_image, (int(a), int(b)), 5, (0, 0, 255), -1)
        else:
            # If no good points found, reinitialize
            rospy.logwarn("No good feature points found, reinitializing")
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.obstacle_detected = False
        
        # Update previous frame and points
        self.prev_gray = gray.copy()
        if len(good_new) > 0:
            self.prev_points = good_new.reshape(-1, 1, 2)
        else:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
        return flow_image

    def lane_follow(self):
        rospy.loginfo("Starting lane following with obstacle detection...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(10)
        self.prev_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            # Check if target distance is reached
            if avg_distance >= self.TARGET_DISTANCE:
                self.stop()
                self.set_led_color('GREEN')
                rospy.loginfo("Target distance reached!")
                break
                
            # Handle obstacle detection
            if self.obstacle_detected:
                self.stop()
                self.set_led_color('RED')
                rospy.logwarn("Obstacle detected, stopping for 2 seconds!")
                # Stop for 2 seconds
                rospy.sleep(2.0)
                # Wait until obstacle is cleared
                while self.obstacle_detected and not rospy.is_shutdown():
                    rospy.loginfo("Waiting for obstacle to clear...")
                    rate.sleep()
                # Obstacle cleared, resume operation
                self.set_led_color('CYAN')
                rospy.loginfo("Obstacle cleared, resuming lane following...")
                self.prev_time = rospy.get_time()  # Reset timing for smooth PD control
                continue
                    
            try:
                lane_msg = rospy.wait_for_message(f"/{self.vehicle_name}/lane_center", Float64, timeout=1.0)
                lane_center = lane_msg.data
                image_center = 320
                
                error = image_center - lane_center
                current_time = rospy.get_time()
                dt = current_time - self.prev_time if self.prev_time is not None else 0.0
                error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                
                omega = (self.KP * error) + (self.KD * error_derivative)
                omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                
                cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                self.pub_cmd.publish(cmd)
                
                self.prev_error = error
                self.prev_time = current_time
            except rospy.ROSException:
                cmd = Twist2DStamped(v=self.VELOCITY/2, omega=0.0)
                self.pub_cmd.publish(cmd)
                rospy.logwarn("Lane center not detected, moving straight slowly")
                
            rate.sleep()

    def detect_lanes_birdseye(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply perspective transform
        birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
        # Find lane center using edge detection
        # Sum the pixel intensities horizontally in the lower half of the image
        roi = birdseye[int(birdseye.shape[0]/2):, :]
        horizontal_sum = np.sum(roi, axis=0)
        
        # Find the center of mass of the edges
        if np.sum(horizontal_sum) > 1000:  # Threshold to ensure we have enough edges
            lane_center = np.average(np.arange(len(horizontal_sum)), weights=horizontal_sum)
            return lane_center
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
        colors = {
            'GREEN': ColorRGBA(0, 1, 0, 1),
            'RED': ColorRGBA(1, 0, 0, 1),
            'CYAN': ColorRGBA(0, 1, 1, 1)
        }
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

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.stop()
        super(DShapeNode, self).on_shutdown()

    def signal_handler(self, sig, frame):
        rospy.loginfo("Ctrl+C detected, shutting down...")
        self.on_shutdown()
        sys.exit(0)

    def run(self):
        rospy.sleep(0.5)
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

# class DShapeNode(DTROS):
#     def __init__(self, node_name):
#         super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         self.vehicle_name = os.environ['VEHICLE_NAME']
        
#         self.camera_matrix = None
#         self.distortion_coeffs = None
        
#         # Homography parameters
#         self.src_points = np.float32([[50, 400], [620, 400], [360, 180], [250, 180]])
#         self.dst_points = np.float32([[180, 500], [440, 500], [440, 0], [180, 0]])
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
        
#         # Optical flow parameters
#         self.prev_gray = None
#         self.prev_points = None
#         self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
#                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
#         # Publishers
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
#         self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
#         self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
#         self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
#         self.pub_flow = rospy.Publisher(f"/{self.vehicle_name}/optical_flow/image/compressed", CompressedImage, queue_size=1)

#         # Subscribers
#         self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)
#         self.sub_camera_info = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/camera_info", CameraInfo, self.cb_camera_info)

#         # Image processing variables
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

#     def cb_camera(self, msg):
#         if self.camera_matrix is None or self.distortion_coeffs is None:
#             return
            
#         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
#         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)

#         # Optical flow processing
#         flow_magnitude = None
#         if self.prev_gray is not None and self.prev_points is not None:
#             next_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None, **self.lk_params)
#             good_new = next_points[status == 1]
#             good_old = self.prev_points[status == 1]
#             flow_vis = undistorted_image.copy()
#             for i, (new, old) in enumerate(zip(good_new, good_old)):
#                 a, b = new.ravel()
#                 c, d = old.ravel()
#                 flow_vis = cv2.line(flow_vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#                 flow_vis = cv2.circle(flow_vis, (int(a), int(b)), 5, (0, 0, 255), -1)
#             flow_magnitude = np.sqrt((good_new - good_old) ** 2).sum(axis=1)
#             msg_out = self.bridge.cv2_to_compressed_imgmsg(flow_vis)
#             msg_out.header = msg.header
#             self.pub_flow.publish(msg_out)
        
#         self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#         self.prev_gray = gray.copy()

#         # Lane detection
#         image_with_points = self.draw_points(undistorted_image, self.src_points)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blurred, 50, 150)
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
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

#         # Return flow magnitude for obstacle detection
#         return flow_magnitude

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
#         self._left_distance_traveled += (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC

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
#         self._right_distance_traveled += (delta_ticks / float(self.TICKS_PER_REV)) * self.WHEEL_CIRC

#     def set_led_color(self, color):
#         pattern = LEDPattern()
#         colors = {'GREEN': ColorRGBA(0, 1, 0, 1), 'RED': ColorRGBA(1, 0, 0, 1), 'CYAN': ColorRGBA(0, 1, 1, 1)}
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

#     def avoid_obstacle(self):
#         rospy.loginfo("Obstacle detected within 0.1m! Initiating avoidance maneuver...")
#         self.set_led_color('RED')
        
#         # Stop for 3 seconds
#         self.stop()
#         rospy.sleep(3.0)
        
#         # Turn left (90 degrees)
#         turn_time = (math.pi / 2) / self.angular_vel  # Time = angle / angular velocity
#         cmd = Twist2DStamped(v=0.0, omega=self.angular_vel)
#         self.pub_cmd.publish(cmd)
#         rospy.sleep(turn_time)
#         self.stop()
        
#         # Drive forward to clear obstacle (e.g., 0.3 meters)
#         forward_time = 0.3 / self.VELOCITY
#         cmd = Twist2DStamped(v=self.VELOCITY, omega=0.0)
#         self.pub_cmd.publish(cmd)
#         rospy.sleep(forward_time)
#         self.stop()
        
#         # Turn right to realign with lane (90 degrees)
#         cmd = Twist2DStamped(v=0.0, omega=-self.angular_vel)
#         self.pub_cmd.publish(cmd)
#         rospy.sleep(turn_time)
#         self.stop()
        
#         self.set_led_color('CYAN')
#         rospy.loginfo("Obstacle avoidance complete, resuming lane following...")

#     def lane_follow(self):
#         rospy.loginfo("Starting lane following with obstacle detection...")
#         self.set_led_color('CYAN')
        
#         rate = rospy.Rate(10)  # 10 Hz
#         self.prev_time = rospy.get_time()
        
#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
#             if avg_distance >= self.TARGET_DISTANCE:
#                 self.stop()
#                 self.set_led_color('GREEN')
#                 rospy.loginfo("Target distance reached!")
#                 break
                
#             # Check for obstacles using optical flow
#             flow_magnitude = self.cb_camera(rospy.wait_for_message(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, timeout=1.0))
#             if flow_magnitude is not None and np.any(flow_magnitude > 10):  # Threshold for obstacle detection
#                 # Rough distance estimation (calibration needed)
#                 # Assuming flow magnitude > 20 corresponds to < 0.1m (tune this)
#                 self.avoid_obstacle()
#                 continue

#             # Lane following with PD control
#             try:
#                 lane_msg = rospy.wait_for_message(f"/{self.vehicle_name}/lane_center", Float64, timeout=1.0)
#                 lane_center = lane_msg.data
#                 image_center = 320
                
#                 error = image_center - lane_center
#                 current_time = rospy.get_time()
#                 dt = current_time - self.prev_time if self.prev_time is not None else 0.0
#                 error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                
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
        
#         self.camera_matrix = None
#         self.distortion_coeffs = None
        
#         # Homography parameters
#         self.src_points = np.float32([
#             [50, 400],      # bottom-left
#             [620, 400],     # bottom-right
#             [360, 180],     # top-right
#             [250, 180]      # top-left
#         ])
        
#         self.dst_points = np.float32([
#             [180, 500],  # bottom-left
#             [440, 500],  # bottom-right
#             [440, 0],    # top-right
#             [180, 0]     # top-left
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

#         # Optical flow parameters
#         self.prev_gray = None
#         self.prev_points = None
#         self.lk_params = dict(winSize=(15, 15),
#                             maxLevel=2,
#                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#         self.feature_params = dict(maxCorners=100,
#                                  qualityLevel=0.3,
#                                  minDistance=7,
#                                  blockSize=7)
        
#         # Obstacle avoidance parameters
#         self.OBSTACLE_THRESHOLD = 20  # Flow magnitude threshold
#         self.AVOIDANCE_DURATION = 2.0
#         self.AVOIDANCE_OMEGA = 2.0
#         self.MIN_OBSTACLE_DISTANCE = 0.10  # 5 cm in meters
#         self.FRAME_RATE = 10  # Hz, matches the rate in lane_follow
        
#         # State variables
#         self.obstacle_detected = False
#         self.avoidance_start_time = None

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

#         # Image processing variables
#         self.bridge = CvBridge()
        
#         # Variables for derivative term
#         self.prev_error = 0.0
#         self.prev_time = None
        
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
    
#     def cb_camera_info(self, msg):
#         self.camera_matrix = np.array(msg.K).reshape(3, 3)
#         self.distortion_coeffs = np.array(msg.D)

#     def cb_camera(self, msg):
#         if self.camera_matrix is None or self.distortion_coeffs is None:
#             return
            
#         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)

#         # Draw points on original image for visualization
#         image_with_points = self.draw_points(undistorted_image, self.src_points)
        
#         # Convert to grayscale
#         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Edge detection
#         edges = cv2.Canny(blurred, 50, 150)
        
#         # Apply perspective transform
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
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Apply Gaussian blur
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
#         # Edge detection
#         edges = cv2.Canny(blurred, 50, 150)
        
#         # Apply perspective transform
#         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
#         # Find lane center using edge detection
#         # Sum the pixel intensities horizontally in the lower half of the image
#         roi = birdseye[int(birdseye.shape[0]/2):, :]
#         horizontal_sum = np.sum(roi, axis=0)
        
#         # Find the center of mass of the edges
#         if np.sum(horizontal_sum) > 1000:  # Threshold to ensure we have enough edges
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

#     def process_optical_flow(self, image):
#         """Process optical flow and return True if obstacle is within 5 cm"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         if self.prev_gray is not None and self.prev_points is not None:
#             next_points, status, err = cv2.calcOpticalFlowPyrLK(
#                 self.prev_gray, gray, self.prev_points, None, **self.lk_params)
            
#             good_new = next_points[status == 1]
#             good_old = self.prev_points[status == 1]
            
#             # Calculate flow magnitude (pixels per frame)
#             flow_magnitude = np.sqrt((good_new - good_old) ** 2).sum(axis=1)
            
#             # Estimate distance: distance = velocity * time_to_collision
#             # time_to_collision ≈ (pixel distance to object) / flow_magnitude
#             # Here, we approximate using flow magnitude and known velocity
#             if np.any(flow_magnitude > self.OBSTACLE_THRESHOLD):
#                 # Average flow magnitude for estimation
#                 avg_flow = np.mean(flow_magnitude[flow_magnitude > self.OBSTACLE_THRESHOLD])
#                 # Time to collision (seconds) = pixels moved per frame / frames per second
#                 time_to_collision = 1.0 / (avg_flow * self.FRAME_RATE)
#                 # Distance (meters) = velocity (m/s) * time (s)
#                 estimated_distance = self.VELOCITY * time_to_collision
                
#                 rospy.loginfo(f"Estimated obstacle distance: {estimated_distance:.3f} meters")
                
#                 # Only detect obstacle if within 5 cm
#                 obstacle_detected = estimated_distance < self.MIN_OBSTACLE_DISTANCE
                
#                 self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#                 self.prev_gray = gray.copy()
                
#                 return obstacle_detected
            
#             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#             self.prev_gray = gray.copy()
#             return False
        
#         self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#         self.prev_gray = gray.copy()
#         return False

#     def lane_follow(self):
#         rospy.loginfo("Starting lane following with obstacle avoidance...")
#         self.set_led_color('CYAN')
        
#         rate = rospy.Rate(self.FRAME_RATE)  # 10 Hz
#         self.prev_time = rospy.get_time()
        
#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
#             if avg_distance >= self.TARGET_DISTANCE:
#                 self.stop()
#                 self.set_led_color('GREEN')
#                 rospy.loginfo("Target distance reached!")
#                 break
                
#             try:
#                 # Get camera image for optical flow
#                 camera_msg = rospy.wait_for_message(f"/{self.vehicle_name}/camera_node/image/compressed", 
#                                                   CompressedImage, timeout=1.0)
#                 image = self.bridge.compressed_imgmsg_to_cv2(camera_msg)
#                 undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
                
#                 # Check for obstacles within 5 cm
#                 if self.process_optical_flow(undistorted_image):
#                     self.obstacle_detected = True
#                     self.avoidance_start_time = rospy.get_time()
#                     rospy.logwarn("Obstacle within 5 cm detected! Initiating avoidance maneuver...")
                
#                 # Handle obstacle avoidance
#                 if self.obstacle_detected:
#                     current_time = rospy.get_time()
#                     if current_time - self.avoidance_start_time < self.AVOIDANCE_DURATION:
#                         # Turn right to avoid obstacle
#                         cmd = Twist2DStamped(v=self.VELOCITY/2, omega=self.AVOIDANCE_OMEGA)
#                         self.pub_cmd.publish(cmd)
#                         self.set_led_color('RED')
#                     else:
#                         # Resume normal operation after avoidance
#                         self.obstacle_detected = False
#                         self.set_led_color('CYAN')
#                 else:
#                     # Normal lane following
#                     lane_msg = rospy.wait_for_message(f"/{self.vehicle_name}/lane_center", 
#                                                     Float64, timeout=1.0)
#                     lane_center = lane_msg.data
#                     image_center = 320
                    
#                     error = image_center - lane_center
                    
#                     current_time = rospy.get_time()
#                     if self.prev_time is not None and (current_time - self.prev_time) > 0:
#                         error_derivative = (error - self.prev_error) / (current_time - self.prev_time)
#                     else:
#                         error_derivative = 0.0
                    
#                     omega = (self.KP * error) + (self.KD * error_derivative)
#                     omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                    
#                     cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#                     self.pub_cmd.publish(cmd)
                    
#                     self.prev_error = error
#                     self.prev_time = current_time
                    
#             except rospy.ROSException as e:
#                 rospy.logwarn(f"Data not received: {str(e)}. Moving straight slowly...")
#                 cmd = Twist2DStamped(v=self.VELOCITY/2, omega=0.0)
#                 self.pub_cmd.publish(cmd)
                
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