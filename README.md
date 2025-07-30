# Autonomous Navigation on a Duckiebot
## This project runs on Ubuntu Linux

## Introduction
This project transforms a Duckiebot, an affordable robotics education platform, into an autonomous navigation system. By integrating advanced computer vision techniques, the Duckiebot can follow lanes, avoid obstacles, detect specific markers to stop, and display augmented reality (AR) overlays. Built using the Robot Operating System (ROS), this system processes the Duckiebot’s camera feed in real-time, making it a practical tool for learning about robotics and autonomous driving.

## Project Overview

### Lane Following
The Duckiebot uses a bird’s-eye view transformation to detect and follow lane markings on the road. It identifies the lane center based on edge detection and adjusts its steering with a PD controller to stay on course, ensuring smooth navigation through straight and curved paths.

### AprilTag Detection
The system detects AprilTags—small markers with unique IDs—placed in the environment. When it spots Tag ID 51 at a specific distance, the Duckiebot stops for 2 seconds, indicated by a red LED, then resumes with a cyan LED, demonstrating precise navigation control.

### Obstacle Avoidance
Using optical flow, the Duckiebot tracks motion in its camera feed to detect obstacles. It stops when it senses objects within a close range, effectively avoiding static obstacles like boxes, though it struggles with fast-moving objects.

### Augmented Reality Overlay
An AR feature overlays virtual images onto flat surfaces in the camera view, enhancing the interactive experience. The system identifies suitable surfaces and blends the overlays seamlessly, though performance varies with lighting and surface size.

### Wheel Odometry and Arc Movement
The Duckiebot calculates its movement using wheel encoder data, enabling it to perform a quarter-circle arc maneuver. This odometry-based control helps it navigate turns accurately, though wheel slip can affect precision.

### Navigation Strategy
The project follows a two-phase approach: first, it follows lanes for a set distance and executes an arc turn; then, it continues lane following while avoiding obstacles. This phased navigation showcases a practical autonomous driving sequence.

## Experimental Results
- **Lane Following**: The Duckiebot stayed aligned with lanes, with minor oscillations on sharp curves.
- **AprilTag Detection**: Successfully stopped at Tag ID 51 under good lighting, with occasional misses in low light.
- **Obstacle Avoidance**: Stopped for static obstacles within 0.1 meters in most cases, but struggled with fast-moving objects.
- **Augmented Reality**: Overlays worked well on large, well-lit surfaces but were less stable on small or noisy ones.
- **Arc Movement**: Completed quarter-circle turns effectively, though sensitive to wheel slip.

## Conclusions
This project successfully combines lane following, AprilTag detection, obstacle avoidance, and AR on a Duckiebot, offering a valuable platform for robotics education. Its strengths include robust navigation and interactive features, though it faces challenges with computational demands and dynamic obstacle detection.

### Future Improvements
- Optimize the system for better real-time performance.
- Enhance obstacle detection with denser optical flow methods.
- Improve AR stability with advanced feature matching.
- Add sensors like LiDAR for better obstacle detection.
- Test on more complex environments with intersections.

### Lessons Learned
Balancing computational resources with feature complexity is key, and optical flow works best for static rather than dynamic obstacles.

This Duckiebot project demonstrates the power of integrating multiple vision techniques, providing an engaging and educational tool for autonomous navigation.


## Components:
### 1. Bird eye view of homography estimation
#### https://youtu.be/XEh2SXvbXc0
### 2. Optical Flow Obstacle Avoidance During Run
#### https://youtu.be/LtNKioPx3Mg
### 3. Lane Following with Optical Flow Detection
#### https://youtu.be/chMo7etrS-4
### 4. Use of AR
#### https://youtu.be/c1ViTTHEqv8
