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
        
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Homography parameters (from your CameraReaderNode)
        self.src_points = np.float32([
            [50, 400],      # bottom-left
            [620, 400],     # bottom-right
            [360, 180],     # top-right
            [250, 180]      # top-left
        ])
        
        self.dst_points = np.float32([
            [180, 500],  # bottom-left
            [440, 500],  # bottom-right
            [440, 0],    # top-right
            [180, 0]     # top-left
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
        self.OMEGA_SPEED = 5
        self.angular_vel = 2.6
        
        # Control parameters
        self.KP = 0.0075  # Proportional gain (may need tuning for bird's-eye view)
        self.KD = 0.01    # Derivative gain
        self.TARGET_DISTANCE = 1.35  # meters
        
        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        
        self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
        
        # Publishers (add these visualization publishers)
        self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
        self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
        self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)

        # Subscribers
        self.left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
        self.right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
        self.camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.camera_info_topic = f"/{self.vehicle_name}/camera_node/camera_info"

        
        self.sub_left_enc = rospy.Subscriber(self.left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)
        self.sub_right_enc = rospy.Subscriber(self.right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
        self.sub_camera = rospy.Subscriber(self.camera_topic, CompressedImage, self.cb_camera)
        self.sub_camera_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_camera_info)

        # Image processing variables
        self.bridge = CvBridge()
        
        
        # Variables for derivative term
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

        # Draw points on original image for visualization
        image_with_points = self.draw_points(undistorted_image, self.src_points)
        
        # Convert to grayscale
        gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply perspective transform
        birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
        
        # Publish visualization images
        try:
            original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            
            self.pub_original.publish(original_msg)
            self.pub_edges.publish(edges_msg)
            self.pub_birdseye.publish(birdseye_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing visualization images: {str(e)}")
        

        lane_center = self.detect_lanes_birdseye(undistorted_image)
        
        if lane_center is not None:
            self.lane_center_pub.publish(Float64(lane_center))

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

    def lane_follow(self):
        rospy.loginfo("Starting lane following for 1.35 meters with PD control using bird's-eye view...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(10)  # 10 Hz
        self.prev_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            if avg_distance >= self.TARGET_DISTANCE:
                self.stop()
                self.set_led_color('GREEN')
                rospy.loginfo("Target distance reached!")
                break
                
            # Get latest lane center
            try:
                lane_msg = rospy.wait_for_message(f"/{self.vehicle_name}/lane_center", Float64, timeout=1.0)
                lane_center = lane_msg.data
                image_center = 320  # Center of 640px wide bird's-eye view
                
                # Calculate error
                error = image_center - lane_center
                
                # Calculate derivative term
                current_time = rospy.get_time()
                if self.prev_time is not None:
                    dt = current_time - self.prev_time
                    if dt > 0:
                        error_derivative = (error - self.prev_error) / dt
                    else:
                        error_derivative = 0.0
                else:
                    error_derivative = 0.0
                
                # PD control signal
                omega = (self.KP * error) + (self.KD * error_derivative)
                
                # Bound the angular velocity
                omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                
                # Publish control command
                cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
                self.pub_cmd.publish(cmd)
                
                # Update previous values
                self.prev_error = error
                self.prev_time = current_time
            except rospy.ROSException:
                # If lane center not detected, go straight slowly
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
        rospy.sleep(0.5)
        self.lane_follow()

if __name__ == "__main__":
    node = DShapeNode(node_name="d_shape_node")
    node.run()
    rospy.loginfo("DShapeNode main finished.")







# #!/usr/bin/env python3

# import os
# import rospy
# import rosbag
# from duckietown.dtros import DTROS, NodeType
# from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
# from sensor_msgs.msg import CompressedImage
# import math
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class LaneFollowingTask(DTROS):
#     def __init__(self, node_name):
#         super(LaneFollowingTask, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

#         # Vehicle name and topics
#         self.vehicle_name = os.environ['VEHICLE_NAME']
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self._publisher = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

#         # Encoder subscribers
#         left_encoder_topic = f"/{self.vehicle_name}/left_wheel_encoder_node/tick"
#         right_encoder_topic = f"/{self.vehicle_name}/right_wheel_encoder_node/tick"
#         self._right_encoder_sub = rospy.Subscriber(right_encoder_topic, WheelEncoderStamped, self.cb_right_encoder)
#         self._left_encoder_sub = rospy.Subscriber(left_encoder_topic, WheelEncoderStamped, self.cb_left_encoder)

#         # Camera subscriber
#         camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
#         self._camera_sub = rospy.Subscriber(camera_topic, CompressedImage, self.cb_camera)

#         # Distance tracking
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
#         self._last_left_encoder_ticks = None
#         self._last_right_encoder_ticks = None

#         # Wheel/encoder parameters
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRCUM = 2.0 * math.pi * self.WHEEL_RADIUS

#         # Target distance
#         self.TARGET_DISTANCE = 10.0  # 10 meters

#         # Driving parameters
#         self.VELOCITY = 0.3  # m/s forward speed

#         # PD Controller parameters
#         self.KP = 0.0075  # Proportional gain (tune this)
#         self.KD = 0.01  # Derivative gain (tune this)
#         self.last_error = 0.0  # For derivative calculation
#         self.lane_center = 320  # Assuming image width is 640, center is 320

#         # Camera processing
#         self._bridge = CvBridge()
#         self.src_points = np.float32([[50, 400], [620, 400], [360, 180], [250, 180]])
#         self.dst_points = np.float32([[200, 500], [440, 500], [440, 0], [200, 0]])
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

#         # Rosbag for logging
#         self.bag = rosbag.Bag('csc22911_lane_following_10m.bag', 'w')

#     def cb_left_encoder(self, msg):
#         if self._last_left_encoder_ticks is None:
#             self._last_left_encoder_ticks = msg.data
#             return
#         delta_ticks = msg.data - self._last_left_encoder_ticks
#         if delta_ticks > (self.TICKS_PER_REV / 2):
#             delta_ticks -= self.TICKS_PER_REV
#         elif delta_ticks < -(self.TICKS_PER_REV / 2):
#             delta_ticks += self.TICKS_PER_REV
#         self._last_left_encoder_ticks = msg.data
#         revolutions = float(delta_ticks) / self.TICKS_PER_REV
#         distance = revolutions * self.WHEEL_CIRCUM
#         self._left_distance_traveled += distance

#     def cb_right_encoder(self, msg):
#         if self._last_right_encoder_ticks is None:
#             self._last_right_encoder_ticks = msg.data
#             return
#         delta_ticks = msg.data - self._last_right_encoder_ticks
#         if delta_ticks > (self.TICKS_PER_REV / 2):
#             delta_ticks -= self.TICKS_PER_REV
#         elif delta_ticks < -(self.TICKS_PER_REV / 2):
#             delta_ticks += self.TICKS_PER_REV
#         self._last_right_encoder_ticks = msg.data
#         revolutions = float(delta_ticks) / self.TICKS_PER_REV
#         distance = revolutions * self.WHEEL_CIRCUM
#         self._right_distance_traveled += distance

#     def cb_camera(self, msg):
#         try:
#             # Convert image to OpenCV format
#             image = self._bridge.compressed_imgmsg_to_cv2(msg)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             edges = cv2.Canny(blurred, 50, 150)
#             birdseye = cv2.warpPerspective(edges, self.H, (640, 480))

#             # Lane detection: find the center of the lane
#             lower_half = birdseye[int(birdseye.shape[0]/2):, :]
#             column_sums = np.sum(lower_half, axis=0)
#             detected_center = np.argmax(column_sums)
#             if detected_center == 0 or detected_center == 639:  # Edge case: no lane detected
#                 detected_center = self.lane_center  # Default to center

#             # PD control
#             error = self.lane_center - detected_center  # Positive error means robot is left of center
#             derivative = error - self.last_error
#             omega = self.KP * error + self.KD * derivative
#             self.last_error = error

#             # Limit omega to prevent oversteering
#             omega = max(min(omega, 0.5), -0.5)
#             self.current_omega = omega

#         except Exception as e:
#             rospy.logerr(f"Error processing image: {str(e)}")
#             self.current_omega = 0.0  # Default to straight if image processing fails

#     def stop_robot(self):
#         stop_msg = Twist2DStamped(v=0.0, omega=0.0)
#         self._publisher.publish(stop_msg)
#         self.bag.write(f"/{self.vehicle_name}/car_cmd_switch_node/cmd", stop_msg)

#     def run(self):
#         rospy.sleep(2.0)
#         self.current_omega = 0.0  # Initialize omega

#         # Follow the lane for 10 meters
#         rospy.loginfo("Following lane for 10 meters...")
#         while not rospy.is_shutdown():
#             avg_distance = (self._right_distance_traveled + self._left_distance_traveled) / 2
#             if avg_distance >= self.TARGET_DISTANCE:
#                 rospy.loginfo("Reached 10 meters, stopping...")
#                 break

#             # Publish velocity command with PD-controlled steering
#             cmd_msg = Twist2DStamped(v=self.VELOCITY, omega=self.current_omega)
#             self._publisher.publish(cmd_msg)
#             self.bag.write(f"/{self.vehicle_name}/car_cmd_switch_node/cmd", cmd_msg)
#             rospy.Rate(10).sleep()

#         self.stop_robot()

#     def on_shutdown(self):
#         self.stop_robot()
#         self.bag.close()
#         super(LaneFollowingTask, self).on_shutdown()

# if __name__ == '__main__':
#     node = LaneFollowingTask(node_name='lane_following_task')
#     node.run()