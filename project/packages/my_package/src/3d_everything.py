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
        
        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_params = None
        
        # Homography parameters for lane detection
        self.src_points = np.float32([[50, 500], 
                                      [620, 500], 
                                      [360, 180], 
                                      [250, 180]])
        
        self.dst_points = np.float32([[180, 500], 
                                      [440, 500], 
                                      [440, 0], 
                                      [180, 0]])
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
        
        # Map variables
        self.robot_x = 0.0  # Global x position (meters)
        self.robot_y = 0.0  # Global y position (meters)
        self.robot_theta = 0.0  # Global heading (radians)
        self.trajectory = []  # List of (x, y) points
        self.lane_points = []  # List of lane center points in global coords
        self.events = []  # List of (x, y, label) for key events
        self.map_scale = 100  # Pixels per meter
        self.map_size = (1000, 1000)  # Initial map size in pixels
        self.map = np.ones((*self.map_size, 3), dtype=np.uint8) * 255  # White background
        self.map_offset_x = self.map_size[1] // 2  # Center of map in pixels
        self.map_offset_y = self.map_size[0] // 2
        self.update_counter = 0  # To limit map publishing frequency

        # Parameters
        self.TICKS_PER_REV = 135
        self.WHEEL_RADIUS = 0.0318
        self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
        self.VELOCITY = 0.15
        self.OMEGA_SPEED = 3
        # self.KP = 0.0075
        # self.KD = 0.01
        self.KP = 0.0075
        self.KD = 0.001
        self.TARGET_DISTANCE = 1.3

        # Additional parameters for arc movement
        self.BASELINE = 0.1016  # Distance between wheels in meters (Duckiebot spec)
        self.TOL_ANGLE = 0.0   # Tolerance for angle in radians (set to 0 as in first code)
        self.TOL = 0.08        # Small tolerance for arc completion
        
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=10,
            qualityLevel=0.9,
            minDistance=1,
            blockSize=32
        )
        self.prev_gray = None
        self.prev_points = None
        self.obstacle_detected = False
        self.MIN_OBSTACLE_DISTANCE = 0.1  # meters
        
        # AR setup
        self.detector = cv2.ORB.create(nfeatures=500)
        package_path = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(package_path, "images/apriltag.png")
        self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.overlay_img is None:
            rospy.logerr("Could not load overlay image")
            rospy.signal_shutdown("Overlay image not found")
        else:
            if len(self.overlay_img.shape) == 2:
                self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGR)
            elif self.overlay_img.shape[2] == 4:
                self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGRA2BGR)
            self.overlay_img = cv2.resize(self.overlay_img, (100, 100))
        
        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
        self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
        self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
        self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
        self.pub_ar = rospy.Publisher(f"/{self.vehicle_name}/ar_planar_node/image", Image, queue_size=1)
        self.pub_flow = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/optical_flow", Image, queue_size=1)
        self.pub_map = rospy.Publisher(f"/{self.vehicle_name}/track_map", Image, queue_size=1)
        
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

    def expand_map_if_needed(self):
        """Expand the map if the robot's position approaches the edges."""
        x_min = min([x for x, y in self.trajectory], default=self.robot_x) * self.map_scale
        x_max = max([x for x, y in self.trajectory], default=self.robot_x) * self.map_scale
        y_min = min([y for x, y in self.trajectory], default=self.robot_y) * self.map_scale
        y_max = max([y for x, y in self.trajectory], default=self.robot_y) * self.map_scale
        
        padding = 100  # Extra space in pixels
        new_width = int(max(self.map_size[1], abs(x_min) + abs(x_max) + 2 * padding))
        new_height = int(max(self.map_size[0], abs(y_min) + abs(y_max) + 2 * padding))
        
        if new_width > self.map_size[1] or new_height > self.map_size[0]:
            new_map = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
            self.map_offset_x = new_width // 2
            self.map_offset_y = new_height // 2
            old_h, old_w = self.map.shape[:2]
            y_start = (new_height - old_h) // 2
            x_start = (new_width - old_w) // 2
            new_map[y_start:y_start+old_h, x_start:x_start+old_w] = self.map
            self.map = new_map
            self.map_size = (new_height, new_width)
            rospy.loginfo(f"Expanded map to {self.map_size}")

    def update_pose(self):
        """Update robot's global pose based on encoder data."""
        dl = self._left_distance_traveled
        dr = self._right_distance_traveled
        dc = (dl + dr) / 2.0  # Center distance traveled
        dtheta = (dr - dl) / self.BASELINE  # Change in heading
        
        self.robot_theta += dtheta
        self.robot_theta = math.atan2(math.sin(self.robot_theta), math.cos(self.robot_theta))
        self.robot_x += dc * math.cos(self.robot_theta)
        self.robot_y += dc * math.sin(self.robot_theta)
        
        self.trajectory.append((self.robot_x, self.robot_y))
        self._left_distance_traveled = 0.0
        self._right_distance_traveled = 0.0

    def map_lane_center(self):
        """Map the latest lane center to global coordinates."""
        if self.latest_lane_center is not None:
            birdseye_center = np.array([[self.latest_lane_center, 240]], dtype=np.float32)
            H_inv = np.linalg.inv(self.H)
            robot_frame = cv2.perspectiveTransform(birdseye_center[None, :, :], H_inv)[0, 0]
            
            scale = 0.5 / 640.0
            x_robot = (robot_frame[0] - 320) * scale
            y_robot = (480 - robot_frame[1]) * scale
            
            x_global = self.robot_x + (x_robot * math.cos(self.robot_theta) - y_robot * math.sin(self.robot_theta))
            y_global = self.robot_y + (x_robot * math.sin(self.robot_theta) + y_robot * math.cos(self.robot_theta))
            
            self.lane_points.append((x_global, y_global))

    def draw_map(self):
        """Draw the current track map and publish it."""
        self.expand_map_if_needed()
        map_img = self.map.copy()
        
        for i in range(1, len(self.trajectory)):
            x1, y1 = self.trajectory[i-1]
            x2, y2 = self.trajectory[i]
            pt1 = (int(x1 * self.map_scale + self.map_offset_x), int(y1 * self.map_scale + self.map_offset_y))
            pt2 = (int(x2 * self.map_scale + self.map_offset_x), int(y2 * self.map_scale + self.map_offset_y))
            cv2.line(map_img, pt1, pt2, (0, 0, 255), 2)
        
        for x, y in self.lane_points:
            pt = (int(x * self.map_scale + self.map_offset_x), int(y * self.map_scale + self.map_offset_y))
            cv2.circle(map_img, pt, 3, (0, 255, 0), -1)
        
        for x, y, label in self.events:
            pt = (int(x * self.map_scale + self.map_offset_x), int(y * self.map_scale + self.map_offset_y))
            cv2.circle(map_img, pt, 5, (255, 0, 0), -1)
            cv2.putText(map_img, label, (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        robot_pt = (int(self.robot_x * self.map_scale + self.map_offset_x), 
                    int(self.robot_y * self.map_scale + self.map_offset_y))
        cv2.circle(map_img, robot_pt, 5, (0, 0, 255), -1)
        
        map_msg = self.bridge.cv2_to_imgmsg(map_img, encoding="bgr8")
        self.pub_map.publish(map_msg)
        self.map = map_img

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
            self.camera_matrix[0,0],
            self.camera_matrix[1,1],
            self.camera_matrix[0,2],
            self.camera_matrix[1,2]
        )
        # rospy.loginfo("Camera parameters initialized")

    def process_optical_flow(self, image, gray):
        flow_image = image.copy()
        
        if self.prev_gray is None or self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            return flow_image
        
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )
        
        if next_points is None or status is None or len(next_points) == 0:
            rospy.logwarn("Optical flow calculation failed, reinitializing feature points")
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray.copy()
            self.obstacle_detected = False
            return flow_image
        
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]
        
        if len(good_new) > 0 and len(good_old) > 0:
            flow_vectors = good_new - good_old
            magnitudes = np.sqrt(np.sum(flow_vectors**2, axis=1))
            avg_magnitude = np.mean(magnitudes)
            estimated_distance = 1.0 / (avg_magnitude + 1e-6)
            
            if estimated_distance < self.MIN_OBSTACLE_DISTANCE and avg_magnitude > 5.0:
                self.obstacle_detected = True
                rospy.logwarn(f"Obstacle detected at estimated distance: {estimated_distance:.3f}m")
            else:
                self.obstacle_detected = False
                
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                flow_image = cv2.line(flow_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                flow_image = cv2.circle(flow_image, (int(a), int(b)), 5, (0, 0, 255), -1)
        else:
            rospy.logwarn("No good feature points found, reinitializing")
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.obstacle_detected = False
        
        self.prev_gray = gray.copy()
        if len(good_new) > 0:
            self.prev_points = good_new.reshape(-1, 1, 2)
        else:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
        return flow_image

    def find_planar_regions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        largest_region = None
        max_area = 0

        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > 1000 and cv2.isContourConvex(approx) and area > max_area:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cY = int(M["m01"] / M["m00"])
                        if cY > gray.shape[0] * 0.66:
                            continue
                    x, y, w, h = cv2.boundingRect(approx)
                    if w < 10 or h < 10:
                        continue
                    max_area = area
                    largest_region = self.order_points(approx.reshape(4, 2))

        return largest_region

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def compute_homography(self, src_pts, dst_pts):
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            rospy.logerr("Homography computation failed.")
            return None
        return H.astype(np.float32)

    def warp_image(self, image, H, dst_shape):
        warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))
        return warped

    def apply_ar(self, frame):
        largest_region = self.find_planar_regions(frame)
        if largest_region is not None and self.overlay_img is not None:
            h, w = self.overlay_img.shape[:2]
            src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
            H = self.compute_homography(src_pts, largest_region.astype(np.float32))
            
            if H is not None:
                warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
                if warped_overlay.dtype != frame.dtype:
                    warped_overlay = warped_overlay.astype(frame.dtype)
                
                mask = np.zeros_like(frame, dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(largest_region), (255, 255, 255))
                mask = cv2.bitwise_not(mask)
                
                frame_masked = cv2.bitwise_and(frame, mask)
                frame = cv2.bitwise_or(frame_masked, warped_overlay)
        return frame

    def cb_camera(self, msg):
        if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
            rospy.logwarn("Camera parameters not yet initialized")
            return
            
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(msg)
            undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
            image_with_points = self.draw_points(undistorted_image, self.src_points)
            gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
            # Optical flow processing
            flow_image = self.process_optical_flow(undistorted_image, gray)
            
            # AprilTag detection
            self.latest_tags = self.at_detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=0.065
            )
            
            # for tag in self.latest_tags:
            #     if tag.tag_id == 51:
            #         distance = tag.pose_t[2][0]
            #         rospy.loginfo(f"Detected AprilTag 133 at distance: {distance:.3f}m")
            #         for i in range(4):
            #             pt_a = tuple(tag.corners[i].astype(int))
            #             pt_b = tuple(tag.corners[(i + 1) % 4].astype(int))
            #             cv2.line(image_with_points, pt_a, pt_b, (0, 255, 0), 2)
            #         cv2.putText(image_with_points, f"ID: {tag.tag_id}", 
            #                   tuple(tag.corners[0].astype(int)), 
            #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Lane detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
            # Apply AR
            ar_image = self.apply_ar(image_with_points)
            
            # Publish visualization
            original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            ar_msg = self.bridge.cv2_to_imgmsg(ar_image, encoding="bgr8")
            flow_msg = self.bridge.cv2_to_imgmsg(flow_image, encoding="bgr8")
            self.pub_original.publish(original_msg)
            self.pub_edges.publish(edges_msg)
            self.pub_birdseye.publish(birdseye_msg)
            self.pub_ar.publish(ar_msg)
            self.pub_flow.publish(flow_msg)
            
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
    
    def compute_heading(self):
        """Compute the robot's heading based on wheel encoder distances."""
        return (self._right_distance_traveled - self._left_distance_traveled) / self.BASELINE

    def move_arc_quarter_odometry(self, radius, speed, clockwise=False):
        """Perform a quarter-circle arc movement using odometry."""
        # Reset odometry and measure initial heading
        rospy.sleep(0.1)  # Allow encoder callbacks to update
        self._left_distance_traveled = 0.0
        self._right_distance_traveled = 0.0
        self.last_left_ticks = None
        self.last_right_ticks = None
        start_heading = self.compute_heading()
        target_change = math.pi / 2.0  # 90 degrees

        direction = -1.0 if clockwise else 1.0
        omega = direction * (speed / radius)

        # Start publishing velocity commands
        twist = Twist2DStamped()
        twist.v = speed
        twist.omega = omega
        self.set_led_color('RED')  # Indicate arc movement

        rospy.loginfo(
            f"move_arc_quarter_odometry: radius={radius}, speed={speed}, "
            f"start_heading={start_heading:.2f}, clockwise={clockwise}"
        )

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            current_heading = self.compute_heading()
            delta_heading = current_heading - start_heading

            if clockwise:
                if delta_heading <= -((target_change - self.TOL_ANGLE) - self.TOL):
                    self.stop()
                    break
            else:
                if delta_heading >= ((target_change - self.TOL_ANGLE) - self.TOL):
                    self.stop()
                    break

            self.pub_cmd.publish(twist)
            rate.sleep()
        
        self.stop()
        rospy.loginfo("Quarter-circle arc complete (odometry-based).")

    def lane_follow(self):
        rospy.loginfo("Starting lane following with AprilTag detection, AR, and obstacle detection in Phase 2...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(10)
        self.prev_time = rospy.get_time()
        phase = 1
        
        while not rospy.is_shutdown():
            self.update_pose()
            self.map_lane_center()
            
            self.update_counter += 1
            if self.update_counter % 10 == 0:
                self.draw_map()
                self.update_counter = 0
            
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            if avg_distance >= self.TARGET_DISTANCE:
                if phase == 1:
                    self.stop()
                    self.set_led_color('GREEN')
                    rospy.loginfo("First target distance reached! Moving back and turning right...")
                    self.events.append((self.robot_x, self.robot_y, "Phase 1 End"))
                    self.draw_map()
                    
                    self.move_arc_quarter_odometry(radius=0.17, speed=0.5, clockwise=False)
                    rospy.sleep(1.0)
                    self.stop()
                    rospy.loginfo("Maneuver complete! Resetting distance and resuming lane following for 1.5m with optical flow obstacle detection...")
                    self._left_distance_traveled = 0.0
                    self._right_distance_traveled = 0.0
                    self.TARGET_DISTANCE = 0.7
                    self.set_led_color('CYAN')
                    self.draw_map()
                    phase = 2
                    continue

                elif phase == 2:
                    self.stop()
                    self.set_led_color('GREEN')
                    self.events.append((self.robot_x, self.robot_y, "Phase 2 End"))
                    rospy.loginfo("Second target distance (1.0m) reached! Stopping robot completely.")
                    self.draw_map()  # Final map update before exit
                    break
            
            for tag in self.latest_tags:
                if tag.tag_id == 51:
                    distance = tag.pose_t[2][0]
                    if 0.18 <= distance <= 0.20:
                        rospy.loginfo(f"AprilTag 51 detected at {distance:.3f}m, stopping for 2 seconds")
                        self.stop()
                        self.set_led_color('RED')
                        self.events.append((self.robot_x, self.robot_y, "Tag 51"))
                        self.draw_map()
                        rospy.sleep(2.0)
                        self.set_led_color('CYAN')
                        break

            if phase == 2 and self.obstacle_detected:
                self.stop()
                self.set_led_color('RED')
                self.events.append((self.robot_x, self.robot_y, "Obstacle"))
                self.draw_map()
                rospy.logwarn("Obstacle detected in Phase 2, stopping for 2 seconds!")
                rospy.sleep(2.0)
                while self.obstacle_detected and not rospy.is_shutdown():
                    rospy.loginfo("Waiting for obstacle to clear in Phase 2...")
                    rate.sleep()
                self.set_led_color('CYAN')
                rospy.loginfo("Obstacle cleared in Phase 2, resuming lane following...")
                self.prev_time = rospy.get_time()
                self.draw_map()
                continue

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

    def stop(self):
        msg = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd.publish(msg)

    def on_shutdown(self):
        rospy.loginfo("Shutting down node...")
        self.stop()
        self.draw_map()  # Publish final map
        rospy.loginfo(f"Final track map published to '/{self.vehicle_name}/track_map'. View it in rqt_image_view.")
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
# from sensor_msgs.msg import CompressedImage, CameraInfo, Image, PointCloud2
# import sensor_msgs.point_cloud2 as pc2
# import signal
# import sys
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class DShapeNode(DTROS):
#     def __init__(self, node_name):
#         super(DShapeNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
#         self.vehicle_name = os.environ['VEHICLE_NAME']
        
#         # Camera parameters (unchanged)
#         self.camera_matrix = None
#         self.distortion_coeffs = None
#         self.camera_params = None
#         self.fx = 300.0
#         self.fy = 300.0
#         self.cx = 320.0
#         self.cy = 240.0
        
#         # Homography parameters (unchanged)
#         self.src_points = np.float32([[50, 500], [620, 500], [360, 180], [250, 180]])
#         self.dst_points = np.float32([[180, 500], [440, 500], [440, 0], [180, 0]])
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
#         # Encoder variables (unchanged)
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
        
#         # Parameters (unchanged)
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
#         self.VELOCITY = 0.15
#         self.OMEGA_SPEED = 3
#         self.KP = 0.0075
#         self.KD = 0.002
#         self.TARGET_DISTANCE = 1.4
#         self.BASELINE = 0.1016
#         self.TOL_ANGLE = 0.0
#         self.TOL = 0.08
        
#         # Publishers (unchanged)
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
#         self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
#         self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
#         self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
#         self.pub_cloud = rospy.Publisher(f"/{self.vehicle_name}/point_cloud", PointCloud2, queue_size=10)
        
#         # Subscribers (unchanged)
#         self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)
#         self.sub_camera_info = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/camera_info", CameraInfo, self.cb_camera_info)

#         # Image processing variables (unchanged)
#         self.bridge = CvBridge()
#         self.latest_lane_center = None
#         self.cloud_msg = None
        
#         # Control variables (unchanged)
#         self.prev_error = 0.0
#         self.prev_time = None
        
#         # New variables for continuous point cloud recording
#         self.point_cloud_file = f"/tmp/{self.vehicle_name}_pointcloud_continuous.pcd"
#         self.save_interval = 1.0  # Save every 1 second
#         self.last_save_time = rospy.get_time()
#         self.is_first_write = True  # Flag for PCD header
        
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
#         self.camera_params = (
#             self.camera_matrix[0,0],
#             self.camera_matrix[1,1],
#             self.camera_matrix[0,2],
#             self.camera_matrix[1,2]
#         )

#     # def cb_camera(self, msg):
#     #     if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
#     #         rospy.logwarn("Camera parameters not yet initialized")
#     #         return
            
#     #     try:
#     #         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#     #         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
#     #         image_with_points = self.draw_points(undistorted_image, self.src_points)
#     #         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
#     #         # Lane detection
#     #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     #         edges = cv2.Canny(blurred, 50, 150)
#     #         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
#     #         # Point cloud generation
#     #         depth = (gray / 255.0) * 5.0  # Simple depth estimation
#     #         height, width = image.shape[:2]
#     #         downsampled_image = cv2.resize(image, (width//2, height//2))
#     #         downsampled_depth = cv2.resize(depth, (width//2, height//2))
#     #         height, width = downsampled_image.shape[:2]
            
#     #         u, v = np.meshgrid(np.arange(width), np.arange(height))
#     #         x = (u - self.cx/2) * downsampled_depth / self.fx
#     #         y = (v - self.cy/2) * downsampled_depth / self.fy
#     #         z = downsampled_depth
            
#     #         points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     #         colors = downsampled_image.reshape(-1, 3)
            
#     #         cloud_data = []
#     #         for point, color in zip(points, colors):
#     #             x, y, z = point
#     #             r, g, b = color
#     #             rgb = (int(r) << 16) | (int(g) << 8) | int(b)
#     #             cloud_data.append([x, y, z, rgb])
            
#     #         fields = [
#     #             pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
#     #             pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
#     #             pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
#     #             pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
#     #         ]
            
#     #         header = msg.header
#     #         header.frame_id = f"{self.vehicle_name}/camera_frame"
#     #         self.cloud_msg = pc2.create_cloud(header, fields, cloud_data)
#     #         self.pub_cloud.publish(self.cloud_msg)
            
#     #         # Save to file with timestamp
#     #         timestamp = rospy.get_time()
#     #         points = list(pc2.read_points(self.cloud_msg, field_names=("x", "y", "z", "rgb")))
#     #         filename = f"/tmp/{self.vehicle_name}_pointcloud_{timestamp:.2f}.pcd"
#     #         with open(filename, "w") as f:
#     #             f.write("# .PCD v0.7 - Point Cloud Data file format\n")
#     #             f.write("VERSION 0.7\n")
#     #             f.write("FIELDS x y z rgb\n")
#     #             f.write("SIZE 4 4 4 4\n")
#     #             f.write("TYPE F F F U\n")
#     #             f.write("COUNT 1 1 1 1\n")
#     #             f.write(f"WIDTH {len(points)}\n")
#     #             f.write("HEIGHT 1\n")
#     #             f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
#     #             f.write(f"POINTS {len(points)}\n")
#     #             f.write("DATA ascii\n")
#     #             for p in points:
#     #                 x, y, z, rgb = p
#     #                 f.write(f"{x} {y} {z} {rgb}\n")
#     #         rospy.loginfo(f"Saved point cloud to {filename}")
            
#     #         # Publish visualization
#     #         original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#     #         edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#     #         birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#     #         self.pub_original.publish(original_msg)
#     #         self.pub_edges.publish(edges_msg)
#     #         self.pub_birdseye.publish(birdseye_msg)
            
#     #         # Update lane center
#     #         self.latest_lane_center = self.detect_lanes_birdseye(undistorted_image)
#     #         if self.latest_lane_center is not None:
#     #             self.lane_center_pub.publish(Float64(self.latest_lane_center))
                
#     #     except Exception as e:
#     #         rospy.logerr(f"Error in camera callback: {str(e)}")
#     # def cb_camera(self, msg):
#     #     if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
#     #         rospy.logwarn("Camera parameters not yet initialized")
#     #         return
            
#     #     try:
#     #         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#     #         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
#     #         image_with_points = self.draw_points(undistorted_image, self.src_points)
#     #         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
#     #         # Lane detection
#     #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     #         edges = cv2.Canny(blurred, 50, 150)
#     #         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
#     #         # Point cloud generation
#     #         depth = (gray / 255.0) * 5.0  # Simple depth estimation
#     #         height, width = image.shape[:2]
#     #         downsampled_image = cv2.resize(image, (width//2, height//2))
#     #         downsampled_depth = cv2.resize(depth, (width//2, height//2))
#     #         height, width = downsampled_image.shape[:2]
            
#     #         u, v = np.meshgrid(np.arange(width), np.arange(height))
#     #         x = (u - self.cx/2) * downsampled_depth / self.fx
#     #         y = (v - self.cy/2) * downsampled_depth / self.fy
#     #         z = downsampled_depth
            
#     #         points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#     #         colors = downsampled_image.reshape(-1, 3)
            
#     #         cloud_data = []
#     #         for point, color in zip(points, colors):
#     #             x, y, z = point
#     #             r, g, b = color
#     #             rgb = (int(r) << 16) | (int(g) << 8) | int(b)
#     #             cloud_data.append([x, y, z, rgb])
            
#     #         fields = [
#     #             pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
#     #             pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
#     #             pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
#     #             pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
#     #         ]
            
#     #         header = msg.header
#     #         header.frame_id = f"{self.vehicle_name}/camera_frame"
#     #         self.cloud_msg = pc2.create_cloud(header, fields, cloud_data)
#     #         self.pub_cloud.publish(self.cloud_msg)
            
#     #         # Publish visualization
#     #         original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#     #         edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#     #         birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#     #         self.pub_original.publish(original_msg)
#     #         self.pub_edges.publish(edges_msg)
#     #         self.pub_birdseye.publish(birdseye_msg)
            
#     #         # Update lane center
#     #         self.latest_lane_center = self.detect_lanes_birdseye(undistorted_image)
#     #         if self.latest_lane_center is not None:
#     #             self.lane_center_pub.publish(Float64(self.latest_lane_center))
                
#     #     except Exception as e:
#     #         rospy.logerr(f"Error in camera callback: {str(e)}")

#     def cb_camera(self, msg):
#         if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
#             rospy.logwarn("Camera parameters not yet initialized")
#             return
            
#         try:
#             image = self.bridge.compressed_imgmsg_to_cv2(msg)
#             undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
#             image_with_points = self.draw_points(undistorted_image, self.src_points)
#             gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
#             # Lane detection
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             edges = cv2.Canny(blurred, 50, 150)
#             birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
#             # Point cloud generation
#             depth = (gray / 255.0) * 5.0
#             height, width = image.shape[:2]
#             downsampled_image = cv2.resize(image, (width//2, height//2))
#             downsampled_depth = cv2.resize(depth, (width//2, height//2))
#             height, width = downsampled_image.shape[:2]
            
#             u, v = np.meshgrid(np.arange(width), np.arange(height))
#             x = (u - self.cx/2) * downsampled_depth / self.fx
#             y = (v - self.cy/2) * downsampled_depth / self.fy
#             z = downsampled_depth
            
#             points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # Fixed line
#             colors = downsampled_image.reshape(-1, 3)
            
#             cloud_data = []
#             for point, color in zip(points, colors):
#                 x, y, z = point
#                 r, g, b = color
#                 rgb = (int(r) << 16) | (int(g) << 8) | int(b)
#                 cloud_data.append([x, y, z, rgb])
            
#             fields = [
#                 pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
#                 pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
#                 pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
#                 pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
#             ]
            
#             header = msg.header
#             header.frame_id = f"{self.vehicle_name}/camera_frame"
#             self.cloud_msg = pc2.create_cloud(header, fields, cloud_data)
#             self.pub_cloud.publish(self.cloud_msg)
            
#             # Continuous point cloud saving
#             current_time = rospy.get_time()
#             if (current_time - self.last_save_time) >= self.save_interval:
#                 self.save_point_cloud(cloud_data)
#                 self.last_save_time = current_time
            
#             # Publish visualization
#             original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#             edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#             birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#             self.pub_original.publish(original_msg)
#             self.pub_edges.publish(edges_msg)
#             self.pub_birdseye.publish(birdseye_msg)
            
#             # Update lane center
#             self.latest_lane_center = self.detect_lanes_birdseye(undistorted_image)
#             if self.latest_lane_center is not None:
#                 self.lane_center_pub.publish(Float64(self.latest_lane_center))
                
#         except Exception as e:
#             rospy.logerr(f"Error in camera callback: {str(e)}")

#     def save_point_cloud(self, cloud_data):
#         """Append point cloud data to a continuous file."""
#         points = [(p[0], p[1], p[2], p[3]) for p in cloud_data]  # x, y, z, rgb
#         mode = "w" if self.is_first_write else "a"
        
#         with open(self.point_cloud_file, mode) as f:
#             if self.is_first_write:
#                 # Write PCD header
#                 f.write("# .PCD v0.7 - Point Cloud Data file format\n")
#                 f.write("VERSION 0.7\n")
#                 f.write("FIELDS x y z rgb\n")
#                 f.write("SIZE 4 4 4 4\n")
#                 f.write("TYPE F F F U\n")
#                 f.write("COUNT 1 1 1 1\n")
#                 f.write(f"WIDTH {len(points)}\n")  # This will be updated on shutdown
#                 f.write("HEIGHT 1\n")
#                 f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
#                 f.write(f"POINTS {len(points)}\n")  # This will be updated on shutdown
#                 f.write("DATA ascii\n")
#                 self.is_first_write = False
            
#             # Append points
#             for p in points:
#                 x, y, z, rgb = p
#                 f.write(f"{x} {y} {z} {rgb}\n")
        
#         rospy.loginfo(f"Appended {len(points)} points to {self.point_cloud_file}")

#     def save_and_exit(self):
#         """Finalize the continuous point cloud file on shutdown."""
#         if self.cloud_msg:
#             # Append the latest cloud data one last time
#             points = list(pc2.read_points(self.cloud_msg, field_names=("x", "y", "z", "rgb")))
#             with open(self.point_cloud_file, "a") as f:
#                 for p in points:
#                     x, y, z, rgb = p
#                     f.write(f"{x} {y} {z} {rgb}\n")
            
#             # Update WIDTH and POINTS in the header (requires rewriting the file)
#             with open(self.point_cloud_file, "r") as f:
#                 lines = f.readlines()
            
#             total_points = sum(1 for line in lines if len(line.split()) == 4)  # Count data lines
#             lines[6] = f"WIDTH {total_points}\n"
#             lines[8] = f"POINTS {total_points}\n"
            
#             with open(self.point_cloud_file, "w") as f:
#                 f.writelines(lines)
            
#             rospy.loginfo(f"Finalized point cloud with {total_points} points to {self.point_cloud_file}")
        
#         rospy.signal_shutdown("User requested shutdown")

#     def detect_lanes_birdseye(self, image):
#         try:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             edges = cv2.Canny(blurred, 50, 150)
#             birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
#             roi = birdseye[int(birdseye.shape[0]/2):, :]
#             horizontal_sum = np.sum(roi, axis=0)
#             if np.sum(horizontal_sum) > 1000:
#                 return np.average(np.arange(len(horizontal_sum)), weights=horizontal_sum)
#             return None
#         except Exception as e:
#             rospy.logerr(f"Error in lane detection: {str(e)}")
#             return None

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
#         colors = {'GREEN': ColorRGBA(0, 1, 0, 1), 'RED': ColorRGBA(1, 0, 0, 1), 'CYAN': ColorRGBA(0, 1, 1, 1)}
#         selected_color = colors.get(color, ColorRGBA(0.5, 0, 0.5, 1))
#         pattern.color_list = [color] * 5
#         pattern.rgb_vals = [selected_color] * 5
#         pattern.color_mask = [1] * 5
#         pattern.frequency = 1.0
#         pattern.frequency_mask = [1] * 5
#         self.led_pub.publish(pattern)
    
#     def compute_heading(self):
#         return (self._right_distance_traveled - self._left_distance_traveled) / self.BASELINE

#     def move_arc_quarter_odometry(self, radius, speed, clockwise=False):
#         rospy.sleep(0.1)
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         start_heading = self.compute_heading()
#         target_change = math.pi / 2.0

#         direction = -1.0 if clockwise else 1.0
#         omega = direction * (speed / radius)

#         twist = Twist2DStamped()
#         twist.v = speed
#         twist.omega = omega
#         self.set_led_color('RED')

#         rospy.loginfo(f"move_arc_quarter_odometry: radius={radius}, speed={speed}, start_heading={start_heading:.2f}, clockwise={clockwise}")

#         rate = rospy.Rate(100)
#         while not rospy.is_shutdown():
#             current_heading = self.compute_heading()
#             delta_heading = current_heading - start_heading

#             if clockwise:
#                 if delta_heading <= -((target_change - self.TOL_ANGLE) - self.TOL):
#                     self.stop()
#                     break
#             else:
#                 if delta_heading >= ((target_change - self.TOL_ANGLE) - self.TOL):
#                     self.stop()
#                     break

#             self.pub_cmd.publish(twist)
#             rate.sleep()
        
#         self.stop()
#         rospy.loginfo("Quarter-circle arc complete (odometry-based).")

#     def lane_follow(self):
#         rospy.loginfo("Starting lane following...")
#         self.set_led_color('CYAN')
        
#         rate = rospy.Rate(10)
#         self.prev_time = rospy.get_time()
#         phase = 1
        
#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
#             # Check target distance and phase transitions
#             if avg_distance >= self.TARGET_DISTANCE:
#                 if phase == 1:
#                     self.stop()
#                     self.set_led_color('GREEN')
#                     rospy.loginfo("First target distance reached! Performing arc maneuver...")
#                     self.move_arc_quarter_odometry(radius=0.17, speed=0.5, clockwise=False)
#                     rospy.sleep(1.0)
#                     self.stop()
#                     rospy.loginfo("Maneuver complete! Resetting distance and resuming lane following for 1.0m...")
#                     self._left_distance_traveled = 0.0
#                     self._right_distance_traveled = 0.0
#                     self.TARGET_DISTANCE = 0.7
#                     self.set_led_color('CYAN')
#                     phase = 2
#                     continue

#                 elif phase == 2:
#                     self.stop()
#                     self.set_led_color('GREEN')
#                     rospy.loginfo("Second target distance (1.0m) reached! Stopping robot completely.")
#                     break
            
#             # Lane following control
#             if self.latest_lane_center is not None:
#                 image_center = 320
#                 error = image_center - self.latest_lane_center
                
#                 current_time = rospy.get_time()
#                 dt = current_time - self.prev_time if self.prev_time is not None and current_time > self.prev_time else 0.0
#                 error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                
#                 omega = (self.KP * error) + (self.KD * error_derivative)
#                 omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                
#                 cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#                 self.pub_cmd.publish(cmd)
                
#                 self.prev_error = error
#                 self.prev_time = current_time
#             else:
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
#         self.save_and_exit()
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
# from sensor_msgs.msg import CompressedImage, CameraInfo, Image, PointCloud2
# import sensor_msgs.point_cloud2 as pc2
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
        
#         # Camera parameters
#         self.camera_matrix = None
#         self.distortion_coeffs = None
#         self.camera_params = None

#         # Point cloud camera parameters (calibrate these or derive from camera_info)
#         self.fx = 300.0
#         self.fy = 300.0
#         self.cx = 320.0
#         self.cy = 240.0
        
#         # Homography parameters for lane detection
#         self.src_points = np.float32([[50, 500], 
#                                       [620, 500], 
#                                       [360, 180], 
#                                       [250, 180]])
        
#         self.dst_points = np.float32([[180, 500], 
#                                       [440, 500], 
#                                       [440, 0], 
#                                       [180, 0]])
#         self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        
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
        
#         # Encoder variables
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
        
#         # Parameters
#         self.TICKS_PER_REV = 135
#         self.WHEEL_RADIUS = 0.0318
#         self.WHEEL_CIRC = 2.0 * math.pi * self.WHEEL_RADIUS
#         self.VELOCITY = 0.15
#         self.OMEGA_SPEED = 3
#         # self.KP = 0.0075
#         # self.KD = 0.01
#         self.KP = 0.0075
#         self.KD = 0.002
#         self.TARGET_DISTANCE = 1.4

#         # Additional parameters for arc movement
#         self.BASELINE = 0.1016  # Distance between wheels in meters (Duckiebot spec)
#         self.TOL_ANGLE = 0.0   # Tolerance for angle in radians (set to 0 as in first code)
#         self.TOL = 0.08        # Small tolerance for arc completion
        
        
#         # Optical flow parameters
#         self.lk_params = dict(
#             winSize=(15, 15),
#             maxLevel=2,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#         )
#         self.feature_params = dict(
#             maxCorners=10,
#             qualityLevel=0.9,
#             minDistance=1,
#             blockSize=32
#         )
#         self.prev_gray = None
#         self.prev_points = None
#         self.obstacle_detected = False
#         self.MIN_OBSTACLE_DISTANCE = 0.1  # meters
        
#         # AR setup
#         self.detector = cv2.ORB.create(nfeatures=500)
#         package_path = os.path.dirname(os.path.realpath(__file__))
#         image_path = os.path.join(package_path, "images/apriltag.png")
#         self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         if self.overlay_img is None:
#             rospy.logerr("Could not load overlay image")
#             rospy.signal_shutdown("Overlay image not found")
#         else:
#             if len(self.overlay_img.shape) == 2:
#                 self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGR)
#             elif self.overlay_img.shape[2] == 4:
#                 self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGRA2BGR)
#             self.overlay_img = cv2.resize(self.overlay_img, (100, 100))
        
#         # Publishers
#         twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
#         self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
#         self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
#         self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
#         self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
#         self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
#         self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
#         self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
#         self.pub_ar = rospy.Publisher(f"/{self.vehicle_name}/ar_planar_node/image", Image, queue_size=1)
#         self.pub_flow = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/optical_flow", Image, queue_size=1)
#         self.pub_cloud = rospy.Publisher(f"/{self.vehicle_name}/point_cloud", PointCloud2, queue_size=10)
        
#         # Subscribers
#         self.sub_left_enc = rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_left_encoder)
#         self.sub_right_enc = rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_right_encoder)
#         self.sub_camera = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.cb_camera)
#         self.sub_camera_info = rospy.Subscriber(f"/{self.vehicle_name}/camera_node/camera_info", CameraInfo, self.cb_camera_info)

#         # Image processing variables
#         self.bridge = CvBridge()
#         self.latest_lane_center = None
#         self.latest_tags = []
#         self.cloud_msg = None  # Store latest point cloud
        
#         # Control variables
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
#         self.camera_params = (
#             self.camera_matrix[0,0],
#             self.camera_matrix[1,1],
#             self.camera_matrix[0,2],
#             self.camera_matrix[1,2]
#         )
#         # rospy.loginfo("Camera parameters initialized")

#     def process_optical_flow(self, image, gray):
#         flow_image = image.copy()
        
#         if self.prev_gray is None or self.prev_points is None:
#             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#             self.prev_gray = gray.copy()
#             return flow_image
        
#         next_points, status, err = cv2.calcOpticalFlowPyrLK(
#             self.prev_gray, gray, self.prev_points, None, **self.lk_params
#         )
        
#         if next_points is None or status is None or len(next_points) == 0:
#             rospy.logwarn("Optical flow calculation failed, reinitializing feature points")
#             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#             self.prev_gray = gray.copy()
#             self.obstacle_detected = False
#             return flow_image
        
#         good_new = next_points[status == 1]
#         good_old = self.prev_points[status == 1]
        
#         if len(good_new) > 0 and len(good_old) > 0:
#             flow_vectors = good_new - good_old
#             magnitudes = np.sqrt(np.sum(flow_vectors**2, axis=1))
#             avg_magnitude = np.mean(magnitudes)
#             estimated_distance = 1.0 / (avg_magnitude + 1e-6)
            
#             if estimated_distance < self.MIN_OBSTACLE_DISTANCE and avg_magnitude > 5.0:
#                 self.obstacle_detected = True
#                 rospy.logwarn(f"Obstacle detected at estimated distance: {estimated_distance:.3f}m")
#             else:
#                 self.obstacle_detected = False
                
#             for i, (new, old) in enumerate(zip(good_new, good_old)):
#                 a, b = new.ravel()
#                 c, d = old.ravel()
#                 flow_image = cv2.line(flow_image, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#                 flow_image = cv2.circle(flow_image, (int(a), int(b)), 5, (0, 0, 255), -1)
#         else:
#             rospy.logwarn("No good feature points found, reinitializing")
#             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
#             self.obstacle_detected = False
        
#         self.prev_gray = gray.copy()
#         if len(good_new) > 0:
#             self.prev_points = good_new.reshape(-1, 1, 2)
#         else:
#             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
#         return flow_image

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
#                     M = cv2.moments(approx)
#                     if M["m00"] != 0:
#                         cY = int(M["m01"] / M["m00"])
#                         if cY > gray.shape[0] * 0.66:
#                             continue
#                     x, y, w, h = cv2.boundingRect(approx)
#                     if w < 10 or h < 10:
#                         continue
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
#             rospy.logerr("Homography computation failed.")
#             return None
#         return H.astype(np.float32)

#     def warp_image(self, image, H, dst_shape):
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))
#         return warped

#     def apply_ar(self, frame):
#         largest_region = self.find_planar_regions(frame)
#         if largest_region is not None and self.overlay_img is not None:
#             h, w = self.overlay_img.shape[:2]
#             src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
#             H = self.compute_homography(src_pts, largest_region.astype(np.float32))
            
#             if H is not None:
#                 warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
#                 if warped_overlay.dtype != frame.dtype:
#                     warped_overlay = warped_overlay.astype(frame.dtype)
                
#                 mask = np.zeros_like(frame, dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, np.int32(largest_region), (255, 255, 255))
#                 mask = cv2.bitwise_not(mask)
                
#                 frame_masked = cv2.bitwise_and(frame, mask)
#                 frame = cv2.bitwise_or(frame_masked, warped_overlay)
#         return frame
    
#     def cb_camera(self, msg):
#         if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
#             rospy.logwarn("Camera parameters not yet initialized")
#             return
            
#         try:
#             image = self.bridge.compressed_imgmsg_to_cv2(msg)
#             undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
#             image_with_points = self.draw_points(undistorted_image, self.src_points)
#             gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
#             # Optical flow processing
#             flow_image = self.process_optical_flow(undistorted_image, gray)
            
#             # AprilTag detection
#             self.latest_tags = self.at_detector.detect(
#                 gray,
#                 estimate_tag_pose=True,
#                 camera_params=self.camera_params,
#                 tag_size=0.065
#             )
            
#             # Lane detection
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             edges = cv2.Canny(blurred, 50, 150)
#             birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
#             # Apply AR
#             ar_image = self.apply_ar(image_with_points)
            
#             # Point cloud generation
#             depth = (gray / 255.0) * 5.0  # Simple depth estimation
#             height, width = image.shape[:2]
#             downsampled_image = cv2.resize(image, (width//2, height//2))
#             downsampled_depth = cv2.resize(depth, (width//2, height//2))
#             height, width = downsampled_image.shape[:2]
            
#             u, v = np.meshgrid(np.arange(width), np.arange(height))
#             x = (u - self.cx/2) * downsampled_depth / self.fx
#             y = (v - self.cy/2) * downsampled_depth / self.fy
#             z = downsampled_depth
            
#             points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
#             colors = downsampled_image.reshape(-1, 3)
            
#             cloud_data = []
#             for point, color in zip(points, colors):
#                 x, y, z = point
#                 r, g, b = color
#                 rgb = (int(r) << 16) | (int(g) << 8) | int(b)
#                 cloud_data.append([x, y, z, rgb])
            
#             fields = [
#                 pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
#                 pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
#                 pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
#                 pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
#             ]
            
#             header = msg.header
#             header.frame_id = f"{self.vehicle_name}/camera_frame"
#             self.cloud_msg = pc2.create_cloud(header, fields, cloud_data)
#             self.pub_cloud.publish(self.cloud_msg)
            
#             # Publish visualization
#             original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#             edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#             birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#             ar_msg = self.bridge.cv2_to_imgmsg(ar_image, encoding="bgr8")
#             flow_msg = self.bridge.cv2_to_imgmsg(flow_image, encoding="bgr8")
#             self.pub_original.publish(original_msg)
#             self.pub_edges.publish(edges_msg)
#             self.pub_birdseye.publish(birdseye_msg)
#             self.pub_ar.publish(ar_msg)
#             self.pub_flow.publish(flow_msg)
            
#             # Update lane center
#             self.latest_lane_center = self.detect_lanes_birdseye(undistorted_image)
#             if self.latest_lane_center is not None:
#                 self.lane_center_pub.publish(Float64(self.latest_lane_center))
                
#         except Exception as e:
#             rospy.logerr(f"Error in camera callback: {str(e)}")

#     def save_and_exit(self):
#         if self.cloud_msg:
#             points = list(pc2.read_points(self.cloud_msg, field_names=("x", "y", "z", "rgb")))
#             with open(f"/tmp/{self.vehicle_name}_pointcloud.pcd", "w") as f:
#                 f.write("# .PCD v0.7 - Point Cloud Data file format\n")
#                 f.write("VERSION 0.7\n")
#                 f.write("FIELDS x y z rgb\n")
#                 f.write("SIZE 4 4 4 4\n")
#                 f.write("TYPE F F F U\n")
#                 f.write("COUNT 1 1 1 1\n")
#                 f.write(f"WIDTH {len(points)}\n")
#                 f.write("HEIGHT 1\n")
#                 f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
#                 f.write(f"POINTS {len(points)}\n")
#                 f.write("DATA ascii\n")
#                 for p in points:
#                     x, y, z, rgb = p
#                     f.write(f"{x} {y} {z} {rgb}\n")
#             rospy.loginfo(f"Saved point cloud to /tmp/{self.vehicle_name}_pointcloud.pcd")
#         rospy.signal_shutdown("User requested shutdown")
#     # def cb_camera(self, msg):
#     #     if self.camera_matrix is None or self.distortion_coeffs is None or self.camera_params is None:
#     #         rospy.logwarn("Camera parameters not yet initialized")
#     #         return
            
#     #     try:
#     #         image = self.bridge.compressed_imgmsg_to_cv2(msg)
#     #         undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
#     #         image_with_points = self.draw_points(undistorted_image, self.src_points)
#     #         gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
            
#     #         # Optical flow processing
#     #         flow_image = self.process_optical_flow(undistorted_image, gray)
            
#     #         # AprilTag detection
#     #         self.latest_tags = self.at_detector.detect(
#     #             gray,
#     #             estimate_tag_pose=True,
#     #             camera_params=self.camera_params,
#     #             tag_size=0.065
#     #         )
            
#     #         # for tag in self.latest_tags:
#     #         #     if tag.tag_id == 51:
#     #         #         distance = tag.pose_t[2][0]
#     #         #         rospy.loginfo(f"Detected AprilTag 133 at distance: {distance:.3f}m")
#     #         #         for i in range(4):
#     #         #             pt_a = tuple(tag.corners[i].astype(int))
#     #         #             pt_b = tuple(tag.corners[(i + 1) % 4].astype(int))
#     #         #             cv2.line(image_with_points, pt_a, pt_b, (0, 255, 0), 2)
#     #         #         cv2.putText(image_with_points, f"ID: {tag.tag_id}", 
#     #         #                   tuple(tag.corners[0].astype(int)), 
#     #         #                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     #         # Lane detection
#     #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     #         edges = cv2.Canny(blurred, 50, 150)
#     #         birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
            
#     #         # Apply AR
#     #         ar_image = self.apply_ar(image_with_points)
            
#     #         # Publish visualization
#     #         original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
#     #         edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
#     #         birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
#     #         ar_msg = self.bridge.cv2_to_imgmsg(ar_image, encoding="bgr8")
#     #         flow_msg = self.bridge.cv2_to_imgmsg(flow_image, encoding="bgr8")
#     #         self.pub_original.publish(original_msg)
#     #         self.pub_edges.publish(edges_msg)
#     #         self.pub_birdseye.publish(birdseye_msg)
#     #         self.pub_ar.publish(ar_msg)
#     #         self.pub_flow.publish(flow_msg)
            
#     #         # Update lane center
#     #         self.latest_lane_center = self.detect_lanes_birdseye(undistorted_image)
#     #         if self.latest_lane_center is not None:
#     #             self.lane_center_pub.publish(Float64(self.latest_lane_center))
                
#     #     except Exception as e:
#     #         rospy.logerr(f"Error in camera callback: {str(e)}")

#     def detect_lanes_birdseye(self, image):
#         try:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             edges = cv2.Canny(blurred, 50, 150)
#             birdseye = cv2.warpPerspective(edges, self.H, (640, 480))
#             roi = birdseye[int(birdseye.shape[0]/2):, :]
#             horizontal_sum = np.sum(roi, axis=0)
#             if np.sum(horizontal_sum) > 1000:
#                 return np.average(np.arange(len(horizontal_sum)), weights=horizontal_sum)
#             return None
#         except Exception as e:
#             rospy.logerr(f"Error in lane detection: {str(e)}")
#             return None

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
#         colors = {'GREEN': ColorRGBA(0, 1, 0, 1), 'RED': ColorRGBA(1, 0, 0, 1), 'CYAN': ColorRGBA(0, 1, 1, 1)}
#         selected_color = colors.get(color, ColorRGBA(0.5, 0, 0.5, 1))
#         pattern.color_list = [color] * 5
#         pattern.rgb_vals = [selected_color] * 5
#         pattern.color_mask = [1] * 5
#         pattern.frequency = 1.0
#         pattern.frequency_mask = [1] * 5
#         self.led_pub.publish(pattern)
    
#     def compute_heading(self):
#         """Compute the robot's heading based on wheel encoder distances."""
#         return (self._right_distance_traveled - self._left_distance_traveled) / self.BASELINE

#     def move_arc_quarter_odometry(self, radius, speed, clockwise=False):
#         """Perform a quarter-circle arc movement using odometry."""
#         # Reset odometry and measure initial heading
#         rospy.sleep(0.1)  # Allow encoder callbacks to update
#         self._left_distance_traveled = 0.0
#         self._right_distance_traveled = 0.0
#         self.last_left_ticks = None
#         self.last_right_ticks = None
#         start_heading = self.compute_heading()
#         target_change = math.pi / 2.0  # 90 degrees

#         direction = -1.0 if clockwise else 1.0
#         omega = direction * (speed / radius)

#         # Start publishing velocity commands
#         twist = Twist2DStamped()
#         twist.v = speed
#         twist.omega = omega
#         self.set_led_color('RED')  # Indicate arc movement

#         rospy.loginfo(
#             f"move_arc_quarter_odometry: radius={radius}, speed={speed}, "
#             f"start_heading={start_heading:.2f}, clockwise={clockwise}"
#         )

#         rate = rospy.Rate(100)
#         while not rospy.is_shutdown():
#             current_heading = self.compute_heading()
#             delta_heading = current_heading - start_heading

#             if clockwise:
#                 if delta_heading <= -((target_change - self.TOL_ANGLE) - self.TOL):
#                     self.stop()
#                     break
#             else:
#                 if delta_heading >= ((target_change - self.TOL_ANGLE) - self.TOL):
#                     self.stop()
#                     break

#             self.pub_cmd.publish(twist)
#             rate.sleep()
        
#         self.stop()
#         rospy.loginfo("Quarter-circle arc complete (odometry-based).")

#     def lane_follow(self):
#         rospy.loginfo("Starting lane following with AprilTag detection, AR, and obstacle detection in Phase 2...")
#         self.set_led_color('CYAN')
        
#         rate = rospy.Rate(10)
#         self.prev_time = rospy.get_time()
#         phase = 1
        
#         while not rospy.is_shutdown():
#             avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
#             # Check target distance and phase transitions
#             if avg_distance >= self.TARGET_DISTANCE:
#                 if phase == 1:
#                     self.stop()
#                     self.set_led_color('GREEN')
#                     rospy.loginfo("First target distance reached! Moving back and turning right...")

#                     # Perform a quarter-circle arc (clockwise, radius 0.15m, speed 0.6m/s)
#                     self.move_arc_quarter_odometry(radius=0.17, speed=0.5, clockwise=False)
#                     rospy.sleep(1.0)  # Brief pause after arc

#                     # # MOVE BACK
#                     # cmd = Twist2DStamped(v=-self.VELOCITY*1.5, omega=0.0)
#                     # self.pub_cmd.publish(cmd)
#                     # rospy.sleep(1.0)
                    
#                     # # MOVE LEFT
#                     # cmd = Twist2DStamped(v=0.0, omega=self.OMEGA_SPEED*1.1)
#                     # self.pub_cmd.publish(cmd)
#                     # rospy.sleep(1.5)
                    
#                     # # MOVE FORWARD
#                     # cmd = Twist2DStamped(v=self.VELOCITY, omega=0.0)
#                     # self.pub_cmd.publish(cmd)
#                     # rospy.sleep(1.5)

#                     # # MOVE LEFT
#                     # cmd = Twist2DStamped(v=0.0, omega=self.OMEGA_SPEED)
#                     # self.pub_cmd.publish(cmd)
#                     # rospy.sleep(1.3)

#                     # self.stop()
#                     # cmd = Twist2DStamped(v=self.VELOCITY*2, omega=self.OMEGA_SPEED)
#                     # self.pub_cmd.publish(cmd)
#                     # rospy.sleep(1.45)
#                     self.stop()
#                     rospy.loginfo("Maneuver complete! Resetting distance and resuming lane following for 1.5m with optical flow obstacle detection...")
#                     self._left_distance_traveled = 0.0
#                     self._right_distance_traveled = 0.0
#                     self.TARGET_DISTANCE = 1.0
#                     self.set_led_color('CYAN')
#                     phase = 2
#                     continue

#                 elif phase == 2:
#                     self.stop()
#                     self.set_led_color('GREEN')
#                     rospy.loginfo("Second target distance (1.0m) reached! Stopping robot completely.")
#                     break
            
#             # AprilTag handling (active in both phases)
#             for tag in self.latest_tags:
#                 if tag.tag_id == 51:
#                     distance = tag.pose_t[2][0]
#                     if 0.18 <= distance <= 0.20:
#                         rospy.loginfo(f"AprilTag 51 detected at {distance:.3f}m, stopping for 2 seconds")
#                         self.stop()
#                         self.set_led_color('RED')
#                         rospy.sleep(2.0)
#                         self.set_led_color('CYAN')
#                         break
                    

#             # Obstacle detection with optical flow only in phase 2
#             if phase == 2 and self.obstacle_detected:
#                 self.stop()
#                 self.set_led_color('RED')
#                 rospy.logwarn("Obstacle detected in Phase 2, stopping for 2 seconds!")
#                 rospy.sleep(2.0)
#                 while self.obstacle_detected and not rospy.is_shutdown():
#                     rospy.loginfo("Waiting for obstacle to clear in Phase 2...")
#                     rate.sleep()
#                 self.set_led_color('CYAN')
#                 rospy.loginfo("Obstacle cleared in Phase 2, resuming lane following...")
#                 self.prev_time = rospy.get_time()
#                 continue

#             # Lane following control (active in both phases)
#             if self.latest_lane_center is not None:
#                 image_center = 320
#                 error = image_center - self.latest_lane_center
                
#                 current_time = rospy.get_time()
#                 dt = current_time - self.prev_time if self.prev_time is not None and current_time > self.prev_time else 0.0
#                 error_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
                
#                 omega = (self.KP * error) + (self.KD * error_derivative)
#                 omega = max(min(omega, self.OMEGA_SPEED), -self.OMEGA_SPEED)
                
#                 # # Publish control command
#                 cmd = Twist2DStamped(v=self.VELOCITY, omega=omega)
#                 self.pub_cmd.publish(cmd)
                
#                 # Update previous values
#                 self.prev_error = error
#                 self.prev_time = current_time

#             else:
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
#         self.save_and_exit()
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
