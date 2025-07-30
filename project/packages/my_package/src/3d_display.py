#!/usr/bin/env python3

import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_point_cloud_from_bag(bag_file, topic):
    """Load the first PointCloud2 message from a ROS bag file."""
    points = []
    colors = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            # Read points and colors from PointCloud2
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb")):
                x, y, z, rgb = point
                # Extract RGB from packed integer
                r = float((rgb >> 16) & 0xFF) / 255.0
                g = float((rgb >> 8) & 0xFF) / 255.0
                b = float(rgb & 0xFF) / 255.0
                points.append([x, y, z])
                colors.append([r, g, b])
            break  # Only take the first message for simplicity
    
    return np.array(points), np.array(colors)

def visualize_point_cloud(points, colors):
    """Visualize point cloud using Matplotlib."""
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample if too many points for performance (optional)
    if points.shape[0] > 10000:  # Adjust threshold as needed
        indices = np.random.choice(points.shape[0], 10000, replace=False)
        points = points[indices]
        colors = colors[indices]
        print(f"Downsampled to {points.shape[0]} points for visualization")
    
    # Plot points with colors
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud from ROS Bag')
    
    # Add a basic coordinate frame (origin lines)
    ax.plot([0, 1], [0, 0], [0, 0], 'r-', label='X-axis')  # Red X-axis
    ax.plot([0, 0], [0, 1], [0, 0], 'g-', label='Y-axis')  # Green Y-axis
    ax.plot([0, 0], [0, 0], [0, 1], 'b-', label='Z-axis')  # Blue Z-axis
    ax.legend()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Replace with your bag file path and topic
    bag_file = "/home/kirilio3/3d.bag"  # e.g., "2025-04-08-12-00-00.bag"
    topic = "/csc22911/point_cloud"  # e.g., "/duckiebot1/point_cloud"
    
    # Load and visualize
    points, colors = load_point_cloud_from_bag(bag_file, topic)
    if points.size > 0:
        print(f"Loaded {points.shape[0]} points")
        visualize_point_cloud(points, colors)
    else:
        print("No points loaded from the bag file.")