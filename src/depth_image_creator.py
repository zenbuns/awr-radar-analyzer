#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import yaml
import os
import time
import traceback
from scipy.spatial import cKDTree
from pathlib import Path

# Define the directory for calibration files relative to this script's location
# or use an absolute path if preferred.
# Assuming the calibration files are in the workspace root for now.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1] # Adjust if src is deeper
INTRINSICS_PATH = WORKSPACE_ROOT / 'intrinsics.yaml'
EXTRINSICS_PATH = WORKSPACE_ROOT / 'extrinsics_cr_pnp0.yaml'

# --- Helper Functions ---

def project_radar_to_image(radar_points_3d, K, D, R, T):
    """Projects 3D points from radar coordinates to 2D image coordinates."""
    if radar_points_3d is None or radar_points_3d.size == 0 or K is None or R is None or T is None:
        return None, None, None
    try:
        # Ensure input arrays are properly formatted
        radar_points_3d = np.asarray(radar_points_3d, dtype=np.float32).reshape(-1, 3)
        R = np.asarray(R, dtype=np.float32).reshape(3, 3)
        T = np.asarray(T, dtype=np.float32).reshape(3, 1)
        K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        D = np.asarray(D, dtype=np.float32).flatten() if D is not None else None

        # Transform points to camera coordinates to get depth and check validity
        points_camera_frame = (R @ radar_points_3d.T + T).T
        depths = points_camera_frame[:, 2]

        # Filter points behind the camera
        valid_depth_mask = depths > 1e-3 # Small epsilon for numerical stability

        if not np.any(valid_depth_mask):
            return None, None, None # No valid points to project

        valid_points_camera_frame = points_camera_frame[valid_depth_mask]
        valid_depths = depths[valid_depth_mask]

        # Use identity transform for projectPoints since we've already transformed the points
        rvec = np.zeros((3,1), np.float32)
        tvec = np.zeros((3,1), np.float32)
        
        # Project valid points using OpenCV with identity transform
        image_points_2d, _ = cv2.projectPoints(valid_points_camera_frame, rvec, tvec, K, D)

        if image_points_2d is not None:
            image_points_2d = image_points_2d.reshape(-1, 2)
            return image_points_2d, valid_depth_mask, valid_depths # Return depths for valid points
        else:
            return None, None, None
    except Exception as e:
        print(f"Error during projection: {e}")
        traceback.print_exc()
        return None, None, None

# --- Node Class ---

class DepthImageCreator(Node):
    def __init__(self):
        super().__init__('depth_image_creator')

        # --- Parameters ---
        self.declare_parameter('image_topic', '/out')
        self.declare_parameter('pcl_topic', 'ti_mmwave/radar_scan_pcl')
        self.declare_parameter('depth_image_topic', '/depth/image_raw')
        self.declare_parameter('intrinsics_path', str(INTRINSICS_PATH))
        self.declare_parameter('extrinsics_path', str(EXTRINSICS_PATH))
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('slop_time', 0.1) # Allowable time difference (seconds) for sync
        self.declare_parameter('depth_fill_method', 'nearest') # 'nearest', 'average', 'none'
        self.declare_parameter('depth_fill_radius', 5) # Radius for nearest neighbor search (pixels)
        self.declare_parameter('max_depth_clip', 100.0) # Max depth value to include (meters)

        self.image_topic = self.get_parameter('image_topic').value
        self.pcl_topic = self.get_parameter('pcl_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.intrinsics_file = self.get_parameter('intrinsics_path').value
        self.extrinsics_file = self.get_parameter('extrinsics_path').value
        self.queue_size = self.get_parameter('queue_size').value
        self.slop_time = self.get_parameter('slop_time').value
        self.depth_fill_method = self.get_parameter('depth_fill_method').value
        self.depth_fill_radius = self.get_parameter('depth_fill_radius').value
        self.max_depth_clip = self.get_parameter('max_depth_clip').value


        # --- Calibration Data ---
        self.camera_matrix = None
        self.dist_coeffs = None
        self.extrinsic_R = None
        self.extrinsic_T = None
        self.calibration_loaded = False
        self.load_calibration()

        # --- ROS Communication ---
        self.bridge = CvBridge()

        # Manual sensor data QoS profile for ROS 2 Humble
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers with ApproximateTimeSynchronizer
        self.image_sub = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=qos_profile)
        self.pcl_sub = message_filters.Subscriber(self, PointCloud2, self.pcl_topic, qos_profile=qos_profile)

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.pcl_sub],
            self.queue_size,
            self.slop_time # Max time difference between messages
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        # Publisher for Depth Image
        self.depth_pub = self.create_publisher(Image, self.depth_image_topic, 10)

        self.get_logger().info(f"Depth Image Creator node started.")
        self.get_logger().info(f"  Subscribing to Image: {self.image_topic}")
        self.get_logger().info(f"  Subscribing to PointCloud2: {self.pcl_topic}")
        self.get_logger().info(f"  Publishing Depth Image: {self.depth_image_topic}")
        if not self.calibration_loaded:
            self.get_logger().warn("Calibration files not loaded. Depth image generation inactive.")
        else:
            self.get_logger().info("Calibration files loaded successfully.")

    def load_calibration(self):
        """Load intrinsic and extrinsic calibration files."""
        try:
            # Load Intrinsics
            if os.path.exists(self.intrinsics_file):
                with open(self.intrinsics_file, 'r') as f:
                    intr_data = yaml.safe_load(f)
                self.camera_matrix = np.array(intr_data['camera_matrix'], dtype=np.float64)
                self.dist_coeffs = np.array(intr_data['dist_coeffs'], dtype=np.float64)
                self.get_logger().info(f"Loaded intrinsics from {self.intrinsics_file}")
            else:
                self.get_logger().error(f"Intrinsics file not found at {self.intrinsics_file}")
                return

            # Load Extrinsics
            if os.path.exists(self.extrinsics_file):
                with open(self.extrinsics_file, 'r') as f:
                    extr_data = yaml.safe_load(f)
                self.extrinsic_R = np.array(extr_data['rotation_matrix'], dtype=np.float64)
                self.extrinsic_T = np.array(extr_data['translation_vector'], dtype=np.float64).reshape(3, 1)
                self.get_logger().info(f"Loaded extrinsics from {self.extrinsics_file}")
            else:
                self.get_logger().error(f"Extrinsics file not found at {self.extrinsics_file}")
                return

            self.calibration_loaded = True

        except FileNotFoundError as e:
            self.get_logger().error(f"Calibration file not found: {e}")
        except KeyError as e:
            self.get_logger().error(f"Missing key in calibration file: {e}")
        except Exception as e:
            self.get_logger().error(f"Error loading calibration files: {e}")
            traceback.print_exc()

    def synchronized_callback(self, image_msg, pcl_msg):
        """Callback for synchronized image and point cloud messages."""
        if not self.calibration_loaded:
            # self.get_logger().warn("Skipping frame - calibration not loaded.", throttle_duration_sec=5)
            return

        start_time = time.time()

        try:
            # --- 1. Convert Image ---
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            height, width = cv_image.shape[:2]

            # --- 2. Extract Radar Points ---
            # Check if the point cloud is empty
            if pcl_msg.data:
                try:
                    # Assuming fields 'x', 'y', 'z' exist
                    gen = pc2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=True)
                    radar_points_list = list(gen)
                    
                    # Debug: Check if radar points are being received
                    self.get_logger().info(f"Received {len(radar_points_list)} radar points", throttle_duration_sec=5)
                    
                    if not radar_points_list:
                         self.get_logger().warn("Received empty point cloud after filtering NaNs.", throttle_duration_sec=5)
                         # Publish empty depth image? Or just return?
                         # Let's publish an empty one for consistency
                         depth_image = np.zeros((height, width), dtype=np.float32)
                         self.publish_depth_image(depth_image, image_msg.header)
                         return
                    
                    # Convert structured array to regular float32 array by extracting x, y, z values
                    # This fixes the "Cannot cast array" error
                    points_array = np.array([(p[0], p[1], p[2]) for p in radar_points_list], dtype=np.float32)
                    radar_points_3d = points_array
                    
                    # Debug: Print a few points for verification
                    if len(radar_points_3d) > 0:
                        self.get_logger().info(f"First radar point: {radar_points_3d[0]}", throttle_duration_sec=5)
                        
                except Exception as e:
                    self.get_logger().error(f"Error reading point cloud data: {e}")
                    return
            else:
                self.get_logger().warn("Received point cloud message with empty data field.", throttle_duration_sec=5)
                # Publish empty depth image
                depth_image = np.zeros((height, width), dtype=np.float32)
                self.publish_depth_image(depth_image, image_msg.header)
                return

            # --- 3. Project Radar Points to Image ---
            image_points_2d, valid_depth_mask, valid_depths = project_radar_to_image(
                radar_points_3d,
                self.camera_matrix,
                self.dist_coeffs,
                self.extrinsic_R,
                self.extrinsic_T
            )
            
            # Debug: Check projection success
            if image_points_2d is not None and valid_depths is not None:
                self.get_logger().info(f"Successfully projected {len(image_points_2d)} points", throttle_duration_sec=5)
                if len(image_points_2d) > 0:
                    self.get_logger().info(f"First projected point: {image_points_2d[0]} with depth {valid_depths[0]}", throttle_duration_sec=5)
            
            if image_points_2d is None or valid_depths is None:
                self.get_logger().warn("Projection failed or resulted in no valid points.", throttle_duration_sec=5)
                # Publish empty depth image
                depth_image = np.zeros((height, width), dtype=np.float32)
                self.publish_depth_image(depth_image, image_msg.header)
                return

            # --- 4. Create Depth Image ---
            depth_image = np.zeros((height, width), dtype=np.float32)

            # Filter points based on image bounds and max depth
            u_coords = image_points_2d[:, 0]
            v_coords = image_points_2d[:, 1]

            in_bounds_mask = (0 <= u_coords) & (u_coords < width) & \
                             (0 <= v_coords) & (v_coords < height) & \
                             (valid_depths < self.max_depth_clip) # Clip depth
                             
            # Debug: Check how many points are in bounds of the image
            valid_count = np.sum(in_bounds_mask)
            self.get_logger().info(f"{valid_count} points are within image bounds and depth range", throttle_duration_sec=5)

            valid_indices_in_bounds = np.where(in_bounds_mask)[0]

            if valid_indices_in_bounds.size == 0:
                self.get_logger().warn("No valid points projected within image bounds or under max depth.", throttle_duration_sec=5)
                self.publish_depth_image(depth_image, image_msg.header) # Publish zero image
                return

            u_points = np.round(u_coords[valid_indices_in_bounds]).astype(np.int32)
            v_points = np.round(v_coords[valid_indices_in_bounds]).astype(np.int32)
            depth_values = valid_depths[valid_indices_in_bounds]

            # Assign depth values to the corresponding pixels
            # Handle multiple points projecting to the same pixel (take the minimum depth)
            # Create a temporary high-res grid or use another method if sub-pixel accuracy is needed
            # For simplicity, direct assignment (last point wins) or minimum depth:
            coords = np.vstack((v_points, u_points)).T
            unique_coords, first_indices = np.unique(coords, axis=0, return_index=True)

            # Option 1: Simple assignment (last point wins)
            # depth_image[v_points, u_points] = depth_values

            # Option 2: Assign minimum depth if multiple points hit the same pixel
            min_depth_map = {}
            for i in range(len(v_points)):
                coord = (v_points[i], u_points[i])
                depth = depth_values[i]
                if coord not in min_depth_map or depth < min_depth_map[coord]:
                    min_depth_map[coord] = depth

            # Debug: Check if we have values in the min_depth_map
            self.get_logger().info(f"min_depth_map contains {len(min_depth_map)} coordinates with depths", throttle_duration_sec=5)
            if len(min_depth_map) > 0:
                first_coord = list(min_depth_map.keys())[0]
                self.get_logger().info(f"First coordinate: {first_coord} with depth: {min_depth_map[first_coord]}", throttle_duration_sec=5)

            for (v, u), depth in min_depth_map.items():
                 depth_image[v, u] = depth

            # Debug: Check if depth_image has non-zero values
            non_zero_pixels = np.count_nonzero(depth_image)
            self.get_logger().info(f"Depth image has {non_zero_pixels} non-zero pixels", throttle_duration_sec=5)
            if non_zero_pixels > 0:
                max_depth = np.max(depth_image)
                min_depth_nonzero = np.min(depth_image[depth_image > 0]) if non_zero_pixels > 0 else 0
                self.get_logger().info(f"Depth range: min={min_depth_nonzero:.4f}, max={max_depth:.4f}", throttle_duration_sec=5)

            # --- 5. Optional: Fill Empty Pixels ---
            if self.depth_fill_method != 'none':
                # Find pixels with projected points
                filled_pixels = np.where(depth_image > 0)
                if filled_pixels[0].size > 0:
                    # Find pixels that are empty
                    empty_pixels = np.where(depth_image == 0)
                    if empty_pixels[0].size > 0:

                        filled_coords = np.vstack(filled_pixels).T
                        empty_coords = np.vstack(empty_pixels).T

                        # Build KDTree for fast nearest neighbor search
                        tree = cKDTree(filled_coords)

                        # Find nearest neighbors within the radius
                        distances, indices = tree.query(empty_coords, k=1, distance_upper_bound=self.depth_fill_radius)

                        # Fill empty pixels with the depth of their nearest valid neighbor
                        valid_nn_mask = np.isfinite(distances)
                        empty_indices_to_fill = empty_coords[valid_nn_mask]
                        corresponding_filled_indices = filled_coords[indices[valid_nn_mask]]

                        depth_image[empty_indices_to_fill[:, 0], empty_indices_to_fill[:, 1]] = \
                            depth_image[corresponding_filled_indices[:, 0], corresponding_filled_indices[:, 1]]


            # --- 6. Publish Depth Image ---
            self.publish_depth_image(depth_image, image_msg.header)

            # --- Performance Logging ---
            # end_time = time.time()
            # processing_time = (end_time - start_time) * 1000 # ms
            # self.get_logger().debug(f"Processed frame in {processing_time:.2f} ms. Projected {len(u_points)} valid points.", throttle_duration_sec=2)


        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in synchronized_callback: {e}")
            traceback.print_exc()

    def publish_depth_image(self, depth_image, original_header):
        """Convert numpy depth array to ROS Image message and publish."""
        try:
            # Debug: Check input depth image
            non_zero_count = np.count_nonzero(depth_image)
            self.get_logger().info(f"Publishing depth image with {non_zero_count} non-zero pixels", throttle_duration_sec=5)
            if non_zero_count > 0:
                self.get_logger().info(f"Depth stats (m): min={np.min(depth_image[depth_image > 0]):.4f}, max={np.max(depth_image):.4f}", throttle_duration_sec=5)
            
            # Convert float32 meters depth image to uint16 millimeters for standard depth representation
            depth_image_mm = (depth_image * 1000).astype(np.uint16)
            
            # Debug: Check converted image
            non_zero_mm = np.count_nonzero(depth_image_mm)
            self.get_logger().info(f"Converted depth image (mm) has {non_zero_mm} non-zero pixels", throttle_duration_sec=5)
            if non_zero_mm > 0:
                 self.get_logger().info(f"Depth stats (mm): min={np.min(depth_image_mm[depth_image_mm > 0])}, max={np.max(depth_image_mm)}", throttle_duration_sec=5)

            # Convert numpy array to ROS Image message *first*
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image_mm, encoding="16UC1")
            
            # Set the header *on the message returned by CvBridge*
            depth_msg.header = original_header # Use timestamp and frame_id from original image
            depth_msg.header.frame_id = "camera_depth_frame" # Or link to the original camera frame in TF
            
            # Debug: Check the actual message data before publishing
            try:
                # Convert byte array to numpy array for quick check
                msg_data_np = np.frombuffer(depth_msg.data, dtype=np.uint16)
                non_zero_msg_data = np.count_nonzero(msg_data_np)
                self.get_logger().info(f"Final depth_msg.data has {non_zero_msg_data} non-zero values before publishing.", throttle_duration_sec=5)
            except Exception as e_dbg:
                self.get_logger().warn(f"Could not inspect depth_msg.data: {e_dbg}")

            # Publish the final message
            self.depth_pub.publish(depth_msg)
            self.get_logger().info("Depth message published", throttle_duration_sec=5)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert depth image to ROS message: {e}")
            traceback.print_exc()
        except Exception as e:
            self.get_logger().error(f"Failed to publish depth image: {e}")
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    depth_creator = DepthImageCreator()
    try:
        rclpy.spin(depth_creator)
    except KeyboardInterrupt:
        depth_creator.get_logger().info("Shutting down node...")
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        depth_creator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 