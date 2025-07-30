#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge

class ARPlanarNode(DTROS):
    def __init__(self, node_name):
        super(ARPlanarNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._output_topic = f"/{self._vehicle_name}/ar_planar_node/image/compressed"
        self._bridge = CvBridge()
           
        self.detector = cv2.ORB.create(nfeatures = 500)

        package_path = os.path.dirname(os.path.realpath(__file__))
        image_path = os.path.join(package_path, "images/apriltag.png")
        self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.overlay_img is None:
            self.logerr("Could not load overlay image")
            rospy.signal_shutdown("Overlay image not found")
            return
        
        # Convert overlay image to 3-channel BGR if needed
        if len(self.overlay_img.shape) == 2:  # Grayscale
            self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGR)
        elif self.overlay_img.shape[2] == 4:  # RGBA
            self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGRA2BGR)
        elif self.overlay_img.shape[2] != 3:  # Unexpected channels
            self.logerr(f"Overlay image has unexpected channels: {self.overlay_img.shape[2]}")
            rospy.signal_shutdown("Invalid overlay image format")
            return
        
        # Resize overlay image to a smaller resolution
        self.overlay_img = cv2.resize(self.overlay_img, (100, 100))  # Adjust based on your need

        self.pub = rospy.Publisher(self._output_topic, CompressedImage, queue_size=1)
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback, queue_size=1, buff_size=2**24)
    
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
                if area > 4000 and cv2.isContourConvex(approx) and area > max_area:
                    # rospy.loginfo(f"Area: {area}")
                    # rospy.loginfo(f"Max Area: {max_area}")
                    # ------------------- Option 2: Filter by vertical position -------------------
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        cY = int(M["m01"] / M["m00"])
                        if cY > gray.shape[0] * 0.66:  # Bottom third of the image
                            continue

                    # ------------------- Option 3: Filter by size -------------------
                    x, y, w, h = cv2.boundingRect(approx)
                    rospy.loginfo(f"w: {w}")
                    rospy.loginfo(f"h: {h}")
                    if w < 10 or h < 10:
                        
                        continue
                    # elif w > 3500 or h < 3500:
                    #     continue

                    max_area = area
                    largest_region = self.order_points(approx.reshape(4, 2))

                # elif area > 5000:
                #     continue
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
            self.logerr("Homography computation failed. Skipping this region.")
            return None
        return H.astype(np.float32)  # Ensure type is correct

    def warp_image(self, image, H, dst_shape):
        warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))  # width, height
        return warped

    def callback(self, msg):
        frame = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        largest_region = self.find_planar_regions(frame)
        
        if largest_region is not None:
            h, w = self.overlay_img.shape[:2]
            src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
            H = self.compute_homography(src_pts, largest_region.astype(np.float32))
            
            if H is None:
                return  # Skip if homography fails

            warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
            
            if warped_overlay.dtype != frame.dtype:
                warped_overlay = warped_overlay.astype(frame.dtype)
            
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(largest_region), (255, 255, 255))
            mask = cv2.bitwise_not(mask)
            
            frame_masked = cv2.bitwise_and(frame, mask)
            
            if frame_masked.shape != warped_overlay.shape:
                self.logerr(f"Shape mismatch: frame_masked {frame_masked.shape} vs warped_overlay {warped_overlay.shape}")
                return
            
            frame = cv2.bitwise_or(frame_masked, warped_overlay)
        
        output_msg = self._bridge.cv2_to_compressed_imgmsg(frame)
        output_msg.header = msg.header
        self.pub.publish(output_msg)

if __name__ == '__main__':
    node = ARPlanarNode(node_name='ar_planar_node')
    rospy.spin()






# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class ARPlanarNode(DTROS):
#     def __init__(self, node_name):
#         super(ARPlanarNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._output_topic = f"/{self._vehicle_name}/ar_planar_node/image/compressed"
#         self._bridge = CvBridge()
           
#         self.detector = cv2.ORB.create(nfeatures = 500)

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
#         elif self.overlay_img.shape[2] != 3:  # Unexpected channels
#             self.logerr(f"Overlay image has unexpected channels: {self.overlay_img.shape[2]}")
#             rospy.signal_shutdown("Invalid overlay image format")
#             return
        
#         # Resize overlay image to a smaller resolution
#         self.overlay_img = cv2.resize(self.overlay_img, (100, 100))  # Adjust based on your need

#         self.pub = rospy.Publisher(self._output_topic, CompressedImage, queue_size=1)
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback, queue_size=1, buff_size=2**24)
    
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

#                     # ------------------- Option 2: Filter by vertical position -------------------
#                     M = cv2.moments(approx)
#                     if M["m00"] != 0:
#                         cY = int(M["m01"] / M["m00"])
#                         if cY > gray.shape[0] * 0.66:  # Bottom third of the image
#                             continue

#                     # ------------------- Option 3: Filter by size -------------------
#                     x, y, w, h = cv2.boundingRect(approx)
#                     if w < 10 or h < 10:
#                         continue

#                     max_area = area
#                     largest_region = self.order_points(approx.reshape(4, 2))

#         return largest_region

#     # def find_planar_regions(self, frame):
#     #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#     #     keypoints, descriptors = self.detector.detectAndCompute(gray, None)
#     #     blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#     #     edged = cv2.Canny(blurred, 50, 150)
#     #     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#     #     largest_region = None
#     #     max_area = 0

#     #     for cnt in contours:
#     #         perimeter = cv2.arcLength(cnt, True)
#     #         approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
#     #         if len(approx) == 4:
#     #             area = cv2.contourArea(approx)
#     #             if area > 1000 and cv2.isContourConvex(approx) and area > max_area:
#     #                 max_area = area
#     #                 largest_region = self.order_points(approx.reshape(4, 2))

#     #     return largest_region  # Return only the largest region

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
#             self.logerr("Homography computation failed. Skipping this region.")
#             return None
#         return H.astype(np.float32)  # Ensure type is correct

#     def warp_image(self, image, H, dst_shape):
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))  # width, height
#         return warped

#     def callback(self, msg):
#         frame = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         largest_region = self.find_planar_regions(frame)
        
#         if largest_region is not None:
#             h, w = self.overlay_img.shape[:2]
#             src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
#             H = self.compute_homography(src_pts, largest_region.astype(np.float32))
            
#             if H is None:
#                 return  # Skip if homography fails

#             warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
            
#             if warped_overlay.dtype != frame.dtype:
#                 warped_overlay = warped_overlay.astype(frame.dtype)
            
#             mask = np.zeros_like(frame, dtype=np.uint8)
#             cv2.fillConvexPoly(mask, np.int32(largest_region), (255, 255, 255))
#             mask = cv2.bitwise_not(mask)
            
#             frame_masked = cv2.bitwise_and(frame, mask)
            
#             if frame_masked.shape != warped_overlay.shape:
#                 self.logerr(f"Shape mismatch: frame_masked {frame_masked.shape} vs warped_overlay {warped_overlay.shape}")
#                 return
            
#             frame = cv2.bitwise_or(frame_masked, warped_overlay)
        
#         output_msg = self._bridge.cv2_to_compressed_imgmsg(frame)
#         output_msg.header = msg.header
#         self.pub.publish(output_msg)

# if __name__ == '__main__':
#     node = ARPlanarNode(node_name='ar_planar_node')
#     rospy.spin()










# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class ARPlanarNode(DTROS):
#     def __init__(self, node_name):
#         super(ARPlanarNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._output_topic = f"/{self._vehicle_name}/ar_planar_node/image/compressed"
#         self._bridge = CvBridge()
        
#         self.detector = None
#         try:
#             self.detector = cv2.SIFT.create()
#             self.loginfo("Using SIFT detector")
#         except AttributeError:
#             self.detector = cv2.ORB.create(nfeatures=500)
#             self.logwarn("SIFT not available, using ORB detector")

#         package_path = os.path.dirname(os.path.realpath(__file__))
#         image_path = os.path.join(package_path, "images/apriltag.png")
#         self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         if self.overlay_img is None:
#             self.logerr("Could not load overlay image")
#             rospy.signal_shutdown("Overlay image not found")
#             return
        
#         # Resize overlay image to a smaller resolution
#         self.overlay_img = cv2.resize(self.overlay_img, (100, 100))  # Adjust based on your need

#         if len(self.overlay_img.shape) == 2:  # Grayscale
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGR)
#         elif self.overlay_img.shape[2] == 4:  # RGBA
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGRA2BGR)
#         elif self.overlay_img.shape[2] != 3:  # Unexpected channels
#             self.logerr(f"Overlay image has unexpected channels: {self.overlay_img.shape[2]}")
#             rospy.signal_shutdown("Invalid overlay image format")
#             return
        
#         self.pub = rospy.Publisher(self._output_topic, CompressedImage, queue_size=1)
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback, queue_size=1, buff_size=2**24)


#     def find_planar_regions(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         keypoints, descriptors = self.detector.detectAndCompute(gray, None)
#         blurred = cv2.GaussianBlur(gray, (3, 3), 0)
#         edged = cv2.Canny(blurred, 50, 150)
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         planar_regions = []
#         for cnt in contours:
#             perimeter = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
#             if len(approx) == 4:
#                 area = cv2.contourArea(approx)
#                 if area > 1000 and cv2.isContourConvex(approx):
#                     approx = self.order_points(approx.reshape(4, 2))
#                     # Check if the plane is approximately vertical
#                     if self.is_vertical(approx):
#                         planar_regions.append(approx)
#         return planar_regions

#     def order_points(self, pts):
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect

#     def is_vertical(self, pts):
#         # Calculate the angles of the sides relative to the vertical (y-axis)
#         # pts order: top-left, top-right, bottom-right, bottom-left
#         left_edge = pts[3] - pts[0]  # bottom-left - top-left
#         right_edge = pts[2] - pts[1]  # bottom-right - top-right
        
#         # Compute angles in degrees (atan2 returns radians, convert to degrees)
#         angle_left = np.abs(np.arctan2(left_edge[0], left_edge[1]) * 180 / np.pi)
#         angle_right = np.abs(np.arctan2(right_edge[0], right_edge[1]) * 180 / np.pi)
        
#         # A vertical plane will have edges close to 0° or 180° relative to vertical
#         # Allow some tolerance (e.g., within 30° of vertical)
#         tolerance = 30
#         is_vertical_left = (angle_left < tolerance) or (angle_left > 180 - tolerance)
#         is_vertical_right = (angle_right < tolerance) or (angle_right > 180 - tolerance)
        
#         return is_vertical_left and is_vertical_right

#     def compute_homography(self, src_pts, dst_pts):
#         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         return H

#     def warp_image(self, image, H, dst_shape):
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))  # width, height
#         return warped

#     def callback(self, msg):
#         frame = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         planar_regions = self.find_planar_regions(frame)
        
#         if planar_regions:
#             for region in planar_regions:
#                 h, w = self.overlay_img.shape[:2]
#                 src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
#                 H = self.compute_homography(src_pts, region.astype(np.float32))
                
#                 warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
                
#                 if warped_overlay.dtype != frame.dtype:
#                     warped_overlay = warped_overlay.astype(frame.dtype)
                
#                 mask = np.zeros_like(frame, dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, np.int32(region), (255, 255, 255))
#                 mask = cv2.bitwise_not(mask)
                
#                 frame_masked = cv2.bitwise_and(frame, mask)
                
#                 if frame_masked.shape != warped_overlay.shape:
#                     self.logerr(f"Shape mismatch: frame_masked {frame_masked.shape} vs warped_overlay {warped_overlay.shape}")
#                     continue
                
#                 frame = cv2.bitwise_or(frame_masked, warped_overlay)
        
#         output_msg = self._bridge.cv2_to_compressed_imgmsg(frame)
#         output_msg.header = msg.header
#         self.pub.publish(output_msg)

# if __name__ == '__main__':
#     node = ARPlanarNode(node_name='ar_planar_node')
#     rospy.spin()




# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class ARPlanarNode(DTROS):
#     def __init__(self, node_name):
#         super(ARPlanarNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._output_topic = f"/{self._vehicle_name}/ar_planar_node/image/compressed"
#         self._bridge = CvBridge()
        
#         # self.loginfo(f"OpenCV Version: {cv2.__version__}")
#         try:
#             sift = cv2.SIFT.create()
#             self.loginfo("SIFT is available and working")
#         except AttributeError:
#             self.logerr("SIFT is not available. Check OpenCV installation.")
        
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
#         elif self.overlay_img.shape[2] != 3:  # Unexpected channels
#             self.logerr(f"Overlay image has unexpected channels: {self.overlay_img.shape[2]}")
#             rospy.signal_shutdown("Invalid overlay image format")
#             return
#         # self.loginfo(f"Overlay image shape: {self.overlay_img.shape}, dtype: {self.overlay_img.dtype}")
            
#         self.pub = rospy.Publisher(self._output_topic, CompressedImage, queue_size=1)
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

#     def find_planar_regions(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         try:
#             detector = cv2.SIFT.create()
#             self.loginfo("Using SIFT detector")
#         except AttributeError:
#             detector = cv2.ORB.create()
#             self.logwarn("SIFT not available, using ORB detector")
        
#         keypoints, descriptors = detector.detectAndCompute(gray, None)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edged = cv2.Canny(blurred, 30, 200)
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         planar_regions = []
#         for cnt in contours:
#             perimeter = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
#             if len(approx) == 4:
#                 area = cv2.contourArea(approx)
#                 if area > 1000 and cv2.isContourConvex(approx):
#                     approx = self.order_points(approx.reshape(4, 2))
#                     planar_regions.append(approx)
#         return planar_regions

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
#         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         return H

#     def warp_image(self, image, H, dst_shape):
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))  # width, height
#         return warped

#     def callback(self, msg):
#         frame = self._bridge.compressed_imgmsg_to_cv2(msg)
#         # self.loginfo(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        
#         planar_regions = self.find_planar_regions(frame)
        
#         if planar_regions:
#             for region in planar_regions:
#                 h, w = self.overlay_img.shape[:2]
#                 src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
#                 H = self.compute_homography(src_pts, region.astype(np.float32))
                
#                 warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
#                 # self.loginfo(f"Warped overlay shape: {warped_overlay.shape}, dtype: {warped_overlay.dtype}")
                
#                 if warped_overlay.dtype != frame.dtype:
#                     warped_overlay = warped_overlay.astype(frame.dtype)
#                     # self.loginfo(f"Converted warped_overlay dtype to {warped_overlay.dtype}")
                
#                 mask = np.zeros_like(frame, dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, np.int32(region), (255, 255, 255))
#                 mask = cv2.bitwise_not(mask)
                
#                 frame_masked = cv2.bitwise_and(frame, mask)
#                 # self.loginfo(f"Frame_masked shape: {frame_masked.shape}, dtype: {frame_masked.dtype}")
                
#                 if frame_masked.shape != warped_overlay.shape:
#                     self.logerr(f"Shape mismatch: frame_masked {frame_masked.shape} vs warped_overlay {warped_overlay.shape}")
#                     continue
                
#                 frame = cv2.bitwise_or(frame_masked, warped_overlay)
        
#         output_msg = self._bridge.cv2_to_compressed_imgmsg(frame)
#         output_msg.header = msg.header
#         self.pub.publish(output_msg)

# if __name__ == '__main__':
#     node = ARPlanarNode(node_name='ar_planar_node')
#     rospy.spin()


# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class ARPlanarNode(DTROS):

#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(ARPlanarNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._output_topic = f"/{self._vehicle_name}/ar_planar_node/image/compressed"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
        
#         # Load the overlay image
#         package_path = os.path.dirname(os.path.realpath(__file__))
#         image_path = os.path.join(package_path, "images/apriltag.png")
#         self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         if self.overlay_img is None:
#             self.logerr("Could not load overlay image")
#             rospy.signal_shutdown("Overlay image not found")
#             return
        
#         if len(self.overlay_img.shape) == 2:  # Grayscale image
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGRA)
#         elif self.overlay_img.shape[2] == 3:  # BGR image without alpha
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGR2BGRA)

#         # Subscriber for camera input
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback, queue_size=1)
        
#         # Publisher for processed output
#         self.pub = rospy.Publisher(self._output_topic, CompressedImage, queue_size=1)
        
#         # For tracking between frames (if needed)
#         self.prev_gray = None
#         self.prev_planar_regions = None
        
#         # Check if SIFT is available, otherwise use ORB
#         try:
#             self.detector = cv2.SIFT_create()
#             self.loginfo("Using SIFT detector")
#         except AttributeError:
#             self.detector = cv2.ORB_create()
#             self.loginfo("SIFT not available, falling back to ORB detector")

#     def find_planar_regions(self, frame):
#         """Find four-point planar regions in a frame"""
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edged = cv2.Canny(blurred, 30, 200)
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         planar_regions = []
#         for cnt in contours:
#             perimeter = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
#             if len(approx) == 4:
#                 area = cv2.contourArea(approx)
#                 if area > 1000 and cv2.isContourConvex(approx):
#                     approx = self.order_points(approx.reshape(4, 2))
#                     planar_regions.append(approx)
#         return planar_regions

#     def order_points(self, pts):
#         """Arrange four points in consistent order"""
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect

#     def compute_homography(self, src_pts, dst_pts):
#         """Compute homography matrix between source and destination points"""
#         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         return H

#     def warp_image(self, image, H, dst_shape):
#         """Warp image using homography matrix"""
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))
#         return warped

#     def callback(self, msg):
#         # Convert ROS CompressedImage to OpenCV image
#         frame = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Find planar regions
#         current_planar_regions = self.find_planar_regions(frame)
        
#         # Process regions if found
#         if current_planar_regions:
#             for region in current_planar_regions:
#                 # Define source points (corners of overlay image)
#                 h, w = self.overlay_img.shape[:2]
#                 src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
                
#                 # Compute homography
#                 H = self.compute_homography(src_pts, region.astype(np.float32))
                
#                 # Warp overlay image
#                 warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
                
#                 # Create mask for overlay
#                 mask = np.zeros(frame.shape, dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, np.int32(region), (255, 255, 255))
#                 mask = cv2.bitwise_not(mask)
                
#                 # Combine original frame with warped overlay
#                 frame = cv2.bitwise_and(frame, mask)
#                 frame = cv2.bitwise_or(frame, warped_overlay)
        
#         # Convert back to ROS CompressedImage and publish
#         output_msg = self._bridge.cv2_to_compressed_imgmsg(frame)
#         output_msg.header = msg.header  # Preserve timestamp and frame_id
#         self.pub.publish(output_msg)

#     def on_shutdown(self):
#         """Cleanup on node shutdown"""
#         self.loginfo("Shutting down AR Planar Node")

# if __name__ == '__main__':
#     # Create and start the node
#     node = ARPlanarNode(node_name='ar_planar_node')
#     rospy.spin()





# #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage, Image
# import cv2
# import numpy as np
# from cv_bridge import CvBridge

# class ARPlanarNode(DTROS):
#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(ARPlanarNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
#         self._output_topic = f"/{self._vehicle_name}/ar_planar_node/output"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
        
#         # Load overlay image
#         package_path = os.path.dirname(os.path.realpath(__file__))
#         image_path = os.path.join(package_path, "images/apriltag.png")
#         self.overlay_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         # self.overlay_img = cv2.imread('/path/to/your/overlay_image.png')  # Update this path
#         if self.overlay_img is None:
#             self.logerr("Could not load overlay image")
#             return
        
#         if len(self.overlay_img.shape) == 2:  # Grayscale image
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_GRAY2BGRA)
#         elif self.overlay_img.shape[2] == 3:  # BGR image without alpha
#             self.overlay_img = cv2.cvtColor(self.overlay_img, cv2.COLOR_BGR2BGRA)

#         # Construct subscriber and publisher
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
#         self.pub = rospy.Publisher(self._output_topic, Image, queue_size=1)
        
#         # For tracking between frames
#         self.prev_gray = None
#         self.prev_planar_regions = None

#     def find_planar_regions(self, frame):
#         """Find four-point planar regions in a frame"""
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         detector = cv2.SIFT_create()
#         keypoints, descriptors = detector.detectAndCompute(gray, None)
        
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         edged = cv2.Canny(blurred, 30, 200)
#         contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
#         planar_regions = []
#         for cnt in contours:
#             perimeter = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            
#             if len(approx) == 4:
#                 area = cv2.contourArea(approx)
#                 if area > 1000 and cv2.isContourConvex(approx):
#                     approx = self.order_points(approx.reshape(4, 2))
#                     planar_regions.append(approx)
        
#         return planar_regions

#     def order_points(self, pts):
#         """Arrange four points in consistent order"""
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]
#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect

#     def compute_homography(self, src_pts, dst_pts):
#         """Compute homography matrix"""
#         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         return H

#     def warp_image(self, image, H, dst_shape):
#         """Warp image using homography matrix"""
#         warped = cv2.warpPerspective(image, H, (dst_shape[1], dst_shape[0]))
#         return warped

#     def callback(self, msg):
#         # Convert ROS CompressedImage to OpenCV image
#         frame = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Find planar regions
#         current_planar_regions = self.find_planar_regions(frame)
        
#         # Process detected regions
#         if current_planar_regions:
#             for region in current_planar_regions:
#                 # Define source points from overlay image
#                 h, w = self.overlay_img.shape[:2]
#                 src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
                
#                 # Compute homography
#                 H = self.compute_homography(src_pts, region.astype(np.float32))
                
#                 # Warp overlay image
#                 warped_overlay = self.warp_image(self.overlay_img, H, frame.shape[:2])
                
#                 # Create mask for overlay
#                 mask = np.zeros(frame.shape, dtype=np.uint8)
#                 cv2.fillConvexPoly(mask, np.int32(region), (255, 255, 255))
#                 mask = cv2.bitwise_not(mask)
                
#                 # Combine frame with warped overlay
#                 frame = cv2.bitwise_and(frame, mask)
#                 frame = cv2.bitwise_or(frame, warped_overlay)
        
#         # Convert processed frame to ROS Image message and publish
#         output_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
#         output_msg.header = msg.header  # Preserve timestamp and frame_id from input
#         self.pub.publish(output_msg)

# if __name__ == '__main__':
#     # Create and start the node
#     node = ARPlanarNode(node_name='ar_planar_node')
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

# class CameraReaderNode(DTROS):

#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
        
#         # Load the overlay image (e.g., a PNG with transparency)
#         package_path = os.path.dirname(os.path.realpath(__file__))
#         image_path = os.path.join(package_path, "images/apriltag.png")
#         self.overlay_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

#         # self.overlay_image = cv2.imread("/home/kirilio3/Downloads/apriltag.png", cv2.IMREAD_UNCHANGED)  # Use a PNG with alpha channel
        
#         if self.overlay_image is None:
#             rospy.logerr("Failed to load overlay image!")
#             return
        
#         # Resize overlay image if needed (e.g., to 100x100 pixels)
#         self.overlay_image = cv2.resize(self.overlay_image, (100, 100), interpolation=cv2.INTER_AREA)
        
#         if len(self.overlay_image.shape) == 2:  # Grayscale image
#             self.overlay_image = cv2.cvtColor(self.overlay_image, cv2.COLOR_GRAY2BGRA)
#         elif self.overlay_image.shape[2] == 3:  # BGR image without alpha
#             self.overlay_image = cv2.cvtColor(self.overlay_image, cv2.COLOR_BGR2BGRA)

#         # Predefined 2D position for the overlay (x, y)
#         self.overlay_x = 200  # Adjust as needed
#         self.overlay_y = 150  # Adjust as needed
        
#         # Subscriber to camera topic
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Publisher for the modified image
#         self.pub = rospy.Publisher(f"/{self._vehicle_name}/camera_node/image_with_overlay/compressed", 
#                                   CompressedImage, queue_size=10)

#     def overlay_image_on_frame(self, background, overlay, x, y):
#         """Overlay an image with transparency onto the background frame."""
#         # Get dimensions of the overlay image
#         overlay_h, overlay_w = overlay.shape[:2]
        
#         # Ensure the overlay fits within the background
#         if x + overlay_w > background.shape[1] or y + overlay_h > background.shape[0]:
#             rospy.logwarn("Overlay position out of bounds, adjusting...")
#             x = min(x, background.shape[1] - overlay_w)
#             y = min(y, background.shape[0] - overlay_h)
        
#         # Extract the region of interest (ROI) from the background
#         roi = background[y:y + overlay_h, x:x + overlay_w]
        
#         # Split the overlay into color and alpha channels
#         if overlay.shape[2] == 4:                   # Check if image has alpha channel
#             overlay_color = overlay[:, :, :3]       # BGR channels
#             overlay_alpha = overlay[:, :, 3]        # Alpha channel
#             alpha_mask = overlay_alpha / 255.0      # Normalize alpha to [0, 1]
            
#             # Blend the overlay with the background
#             for c in range(3):  # For each color channel
#                 roi[:, :, c] = (1.0 - alpha_mask) * roi[:, :, c] + alpha_mask * overlay_color[:, :, c]
        
#         # Place the modified ROI back into the background
#         background[y:y + overlay_h, x:x + overlay_w] = roi
#         return background

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Overlay the image at the predefined position
#         image_with_overlay = self.overlay_image_on_frame(image, self.overlay_image, self.overlay_x, self.overlay_y)
        
#         # Convert back to CompressedImage message
#         msg_out = self._bridge.cv2_to_compressed_imgmsg(image_with_overlay)
#         msg_out.header = msg.header  # Preserve the original header
        
#         # Publish the modified image
#         self.pub.publish(msg_out)

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     rospy.spin()





#     #!/usr/bin/env python3

# import os
# import rospy
# from duckietown.dtros import DTROS, NodeType
# from sensor_msgs.msg import CompressedImage
# import cv2
# from cv_bridge import CvBridge
# import numpy as np

# class CameraReaderNode(DTROS):

#     def __init__(self, node_name):
#         # Initialize the DTROS parent class
#         super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
#         # Static parameters
#         self._vehicle_name = os.environ['VEHICLE_NAME']
#         self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
#         # Bridge between OpenCV and ROS
#         self._bridge = CvBridge()
#         rospy.loginfo(f"Current working directory: {os.getcwd()}")

    
#         # Load the overlay image (e.g., a PNG with transparency)
#         # self.overlay_image = cv2.imread("/home/kirilio3/Downloads/apriltag.png", cv2.IMREAD_UNCHANGED)  # Use a PNG with alpha channel
        
#         if self.overlay_image is None:
#             rospy.logerr("Failed to load overlay image!")
#             return
        
#         # Resize overlay image if needed (e.g., to 100x100 pixels)
#         self.overlay_image = cv2.resize(self.overlay_image, (100, 100), interpolation=cv2.INTER_AREA)
        
#         # Predefined 2D position for the overlay (x, y)
#         self.overlay_x = 200  # Adjust as needed
#         self.overlay_y = 150  # Adjust as needed
        
#         # Subscriber to camera topic
#         self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
#         # Publisher for the modified image
#         self.pub = rospy.Publisher(f"/{self._vehicle_name}/camera_node/image_with_overlay/compressed", 
#                                   CompressedImage, queue_size=10)

#     def overlay_image_on_frame(self, background, overlay, x, y):
#         """Overlay an image with transparency onto the background frame."""
#         # Get dimensions of the overlay image
#         overlay_h, overlay_w = overlay.shape[:2]
        
#         # Ensure the overlay fits within the background
#         if x + overlay_w > background.shape[1] or y + overlay_h > background.shape[0]:
#             rospy.logwarn("Overlay position out of bounds, adjusting...")
#             x = min(x, background.shape[1] - overlay_w)
#             y = min(y, background.shape[0] - overlay_h)
        
#         # Extract the region of interest (ROI) from the background
#         roi = background[y:y + overlay_h, x:x + overlay_w]
        
#         # Split the overlay into color and alpha channels
#         if overlay.shape[2] == 4:                   # Check if image has alpha channel
#             overlay_color = overlay[:, :, :3]       # BGR channels
#             overlay_alpha = overlay[:, :, 3]        # Alpha channel
#             alpha_mask = overlay_alpha / 255.0      # Normalize alpha to [0, 1]
            
#             # Blend the overlay with the background
#             for c in range(3):  # For each color channel
#                 roi[:, :, c] = (1.0 - alpha_mask) * roi[:, :, c] + alpha_mask * overlay_color[:, :, c]
        
#         # Place the modified ROI back into the background
#         background[y:y + overlay_h, x:x + overlay_w] = roi
#         return background

#     def callback(self, msg):
#         # Convert JPEG bytes to CV image
#         image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
#         # Overlay the image at the predefined position
#         image_with_overlay = self.overlay_image_on_frame(image, self.overlay_image, self.overlay_x, self.overlay_y)
        
#         # Convert back to CompressedImage message
#         msg_out = self._bridge.cv2_to_compressed_imgmsg(image_with_overlay)
#         msg_out.header = msg.header  # Preserve the original header
        
#         # Publish the modified image
#         self.pub.publish(msg_out)

# if __name__ == '__main__':
#     # Create the node
#     node = CameraReaderNode(node_name='camera_reader_node')
#     # Keep spinning
#     rospy.spin()