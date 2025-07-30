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
        
        # Homography parameters for lane detection
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
        self.TARGET_DISTANCE = 1.4
        
        # AR setup
        self.detector = cv2.ORB.create(nfeatures=500)
        package_path = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(package_path, "images/apriltag.png")
        self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.overlay_img is None:
            rospy.logerr("Could not load overlay image")
            rospy.signal_shutdown("Overlay image not found")
        else:
            # Convert overlay image to 3-channel BGR if needed
            if len(self.overlay_img.shape) == 2:  # Grayscale
                self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGR)
            elif self.overlay_img.shape[2] == 4:  # RGBA
                self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGRA2BGR)
            self.overlay_img = cv2.resize(self.overlay_img, (100, 100))  # Adjust size as needed
        
        # Publishers
        twist_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)
        self.led_topic = f"/{self.vehicle_name}/led_emitter_node/led_pattern"
        self.led_pub = rospy.Publisher(self.led_topic, LEDPattern, queue_size=1)
        self.lane_center_pub = rospy.Publisher(f"/{self.vehicle_name}/lane_center", Float64, queue_size=1)
        self.pub_original = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/original", Image, queue_size=1)
        self.pub_edges = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/edges", Image, queue_size=1)
        self.pub_birdseye = rospy.Publisher(f"/{self.vehicle_name}/camera_reader/birdseye", Image, queue_size=1)
        self.pub_ar = rospy.Publisher(f"/{self.vehicle_name}/ar_planar_node/image", Image, queue_size=1)  # AR output

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
                        if cY > gray.shape[0] * 0.66:  # Bottom third of the image
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
        warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))  # width, height
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
            
            # Apply AR
            ar_image = self.apply_ar(image_with_points)
            
            # Publish visualization
            original_msg = self.bridge.cv2_to_imgmsg(image_with_points, encoding="bgr8")
            edges_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            birdseye_msg = self.bridge.cv2_to_imgmsg(birdseye, encoding="mono8")
            ar_msg = self.bridge.cv2_to_imgmsg(ar_image, encoding="bgr8")
            self.pub_original.publish(original_msg)
            self.pub_edges.publish(edges_msg)
            self.pub_birdseye.publish(birdseye_msg)
            self.pub_ar.publish(ar_msg)
            
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
        rospy.loginfo("Starting lane following with AprilTag detection and AR...")
        self.set_led_color('CYAN')
        
        rate = rospy.Rate(10)
        self.prev_time = rospy.get_time()
        phase = 1
        
        while not rospy.is_shutdown():
            avg_distance = (self._left_distance_traveled + self._right_distance_traveled) / 2
            
            if avg_distance >= self.TARGET_DISTANCE:
                if phase == 1:
                    self.stop()
                    self.set_led_color('GREEN')
                    rospy.loginfo("First target distance reached! Moving back and turning right...")
                    rospy.sleep(2.0)

                    self.stop()
                    cmd = Twist2DStamped(v=-self.VELOCITY, omega=0.0)
                    self.pub_cmd.publish(cmd)
                    rospy.sleep(1.0)

                    self.stop()
                    cmd = Twist2DStamped(v=0.0, omega=self.OMEGA_SPEED)
                    self.pub_cmd.publish(cmd)
                    rospy.sleep(1.5)

                    self.stop()
                    cmd = Twist2DStamped(v=self.VELOCITY*2, omega=self.OMEGA_SPEED)
                    self.pub_cmd.publish(cmd)
                    rospy.sleep(1.45)
                    
                    self.stop()
                    rospy.loginfo("Maneuver complete! Resetting distance and resuming lane following for 1.5m...")
                    self._left_distance_traveled = 0.0
                    self._right_distance_traveled = 0.0
                    self.TARGET_DISTANCE = 1.5
                    self.set_led_color('CYAN')
                    phase = 2
                    continue
                elif phase == 2:
                    self.stop()
                    self._left_distance_traveled = 0.0
                    self._right_distance_traveled = 0.0
                    self.TARGET_DISTANCE = 1.0
                    self.set_led_color('GREEN')
                    rospy.loginfo("Second target distance (1.5m) reached! Stopping robot completely.")
                    break
            
            for tag in self.latest_tags:
                if tag.tag_id == 51:
                # if tag.tag_id == 133:
                    distance = tag.pose_t[2][0]
                    if 0.18 <= distance <= 0.20:
                        rospy.loginfo(f"AprilTag 51 detected at {distance:.3f}m, stopping for 2 seconds")
                        self.stop()
                        self.set_led_color('RED')
                        rospy.sleep(2.0)
                        self.set_led_color('CYAN')
                        break

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
        rospy.sleep(0.5)
        self.lane_follow()

if __name__ == "__main__":
    node = DShapeNode(node_name="d_shape_node")
    node.run()
    rospy.loginfo("DShapeNode main finished.")