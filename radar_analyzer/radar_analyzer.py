#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 node for radar point cloud analysis.

This module contains the RadarPointCloudAnalyzer class which subscribes to
radar point cloud data, processes it, and provides visualization and analysis
capabilities.
"""

import os
import time
import threading
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Sequence, Union

# Add PyQt5 imports for signal emission
from PyQt5.QtCore import QObject, pyqtSignal

import numpy as np
from scipy import stats
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.artist import Artist

# ROS 2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Import local modules
from radar_params import RadarExperimentParams, ExperimentData
from radar_analyzer.processing.data_processor import (
    filter_points_in_circle, 
)
from radar_analyzer.visualization.visualizer import (
    setup_visualization,
    update_plot,
    update_circle_position,
    update_circle_radius,
    save_visualization
)
from radar_analyzer.processing.multi_frame import (
    process_multi_frame_data,
    combine_multi_frames,
    compute_multi_frame_metrics,
    load_latest_multi_frame_metrics
)
from radar_analyzer.utils.ros_bag_handler import (
    play_rosbag,
    record_rosbag,
    stop_rosbag,
    seek_rosbag
)
from radar_analyzer.utils.report_generator import generate_comparison_report


# Create a separate signal class to handle PyQt signals
class RadarAnalyzerSignals(QObject):
    """A dedicated PyQt signal handler class.
    
    This class holds all the signals that need to be emitted from the RadarPointCloudAnalyzer
    to the UI components. Using a separate class avoids multiple inheritance issues.
    """
    # Define PyQt signals for UI updates
    update_playback_position_signal = pyqtSignal(float)  # Normalized position (0.0-1.0)
    data_reset_signal = pyqtSignal()  # Signal to indicate data has been reset
    bag_playback_ended = pyqtSignal()  # Signal to indicate bag playback has ended


# Main analyzer class that inherits only from Node
class RadarPointCloudAnalyzer(Node):
    """
    A ROS 2 node that subscribes to radar point cloud data and processes
    it for visualization, collection, and analysis.
    
    This node integrates with a GUI to control data collection, real-time visualization,
    and final reporting. It provides both scatter plot and heatmap views of the
    radar data, with customizable analysis parameters.
    
    Attributes:
        params: Parameters for the radar experiment.
        experiment_data: Container for collected experiment data.
        config_results: Dictionary storing results for different configurations.
        collecting_data: Flag indicating if data collection is active.
        current_data: Dictionary of current point cloud data arrays.
        data_lock: Thread lock for synchronizing data access.
        viz_components: Dictionary of visualization components for the scatter plot.
        heatmap_viz: Dictionary of visualization components for the heatmap.
        heatmap_data: Numpy array for the persistent heatmap.
        live_heatmap_data: Numpy array for the real-time decaying heatmap.
        visible: Flag indicating if visualization is currently visible.
    """

    def __init__(self) -> None:
        """
        Initialize the RadarPointCloudAnalyzer node.

        Creates subscriptions for point cloud, track marker arrays, and occupancy data.
        Sets up internal data structures for experimentation and visualization.
        Initializes ROS 2 subscriptions and visualization components.
        """
        # Initialize the Node class
        super().__init__(node_name='radar_point_cloud_analyzer')
        
        # Create signals object for PyQt communication
        self.signals = RadarAnalyzerSignals()
        
        # --- Parameters for Calibration Point Processing ---
        self.declare_parameter('buffer_duration_sec', 5.0)
        self.declare_parameter('processing_time_window_sec', 3.0)
        self.declare_parameter('max_range_m', 20.0)
        self.declare_parameter('z_score_threshold', 3.0)
        self.declare_parameter('min_inlier_points', 5)
        self.declare_parameter('use_velocity_filter', False)
        self.declare_parameter('velocity_field_name', 'velocity')
        
        # Get calibration processing parameters
        self.buffer_max_duration_sec = self.get_parameter('buffer_duration_sec').get_parameter_value().double_value
        self.time_window_sec = self.get_parameter('processing_time_window_sec').get_parameter_value().double_value
        self.range_threshold_m = self.get_parameter('max_range_m').get_parameter_value().double_value
        self.z_score_thresh = self.get_parameter('z_score_threshold').get_parameter_value().double_value
        self.min_inlier_points = self.get_parameter('min_inlier_points').get_parameter_value().integer_value
        self.use_velocity_filter = self.get_parameter('use_velocity_filter').get_parameter_value().bool_value
        self.velocity_field = self.get_parameter('velocity_field_name').get_parameter_value().string_value
        # --- End Calibration Params ---
        
        # Make the signal accessible directly from this class for easier use
        self.update_playback_position_signal = self.signals.update_playback_position_signal
        self.params = RadarExperimentParams()
        self.experiment_data = ExperimentData()
        self.config_results = {}
        self.collecting_data = False
        self.collection_start_time = None

        # Memory-optimized containers
        self.current_data = {
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32),
            'z': np.array([], dtype=np.float32),
            'intensities': np.array([], dtype=np.float32),
            'circle_x': np.array([], dtype=np.float32),
            'circle_y': np.array([], dtype=np.float32),
            'circle_intensities': np.array([], dtype=np.float32),
            'circle_indices': np.array([], dtype=np.int32)
        }
        
        # Multi-frame processing containers
        self.frame_buffer = []
        self.frame_count = 0  # Initialize frame count for memory management
        self.combined_frame = {
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32),
            'z': np.array([], dtype=np.float32),
            'intensities': np.array([], dtype=np.float32)
        }
        self.multi_frame_metrics = {}

        # Buffer for storing recent PointCloud2 messages for calibration lookup
        self.point_cloud_buffer = deque()

        # Create a more robust QoS profile for better bag playback compatibility
        self.reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        # Define QoS profile for sensor data (often BEST_EFFORT)
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Keep only the latest for best effort
        )
        
        # Use the sensor QoS profile for the point cloud subscription
        self.pcl_subscription = self.create_subscription(
            PointCloud2, '/ti_mmwave/radar_scan_pcl', self.pcl_callback, self.sensor_qos
        )
        self.track_array_subscription = self.create_subscription(
            MarkerArray, '/ti_mmwave/radar_track_marker_array', 
            self.track_array_callback, self.reliable_qos
        )
        self.occupancy_subscription = self.create_subscription(
            Bool, '/ti_mmwave/radar_occupancy', self.occupancy_callback, self.reliable_qos
        )
        
        # Keep track of point cloud reception for diagnostics
        self.last_pcl_msg_time = None
        self.pcl_msg_count = 0

        self.timer = self.create_timer(1.0, self.timer_callback)

        # Visualization components
        self.viz_components = {
            'fig': None,
            'ax': None,
            'scatter': None,
            'circle_scatter': None,
            'sampling_circle': None,
            'stats_text': None,
            'circle_stats_text': None
        }

        # Heatmap visualization
        self.heatmap_viz = {
            'fig': None,
            'ax': None,
            'heatmap': None,
            'colorbar': None,
            'norm': None,
            'contour': None,
            'contour_levels': 6,
            'snr_text': None,
            'roi_indicators': []
        }

        # Cached circle center for performance - initialize with safe defaults
        self._cached_circle_center = np.array([0, self.params.circle_distance])

        # Thread safety
        self.data_lock = threading.Lock()

        # Rate-limiting for visualization updates
        self.last_update_time = time.time()
        self.update_interval = 0.1

        # Visibility flag for optimization
        self.visible = False
        
        # Animation objects
        self.anim = None
        self.heatmap_anim = None
        
        # ROS2 bag recording and playback attributes
        self.rosbag_proc = None
        self.is_recording = False
        self.is_playing = False
        self.current_bag_path = None
        self.bag_start_time = None
        self.bag_duration = 0.0

        # Check ROS 2 availability
        self.ros2_available = self._check_ros2_availability()
        if self.ros2_available:
            self.get_logger().info("ROS 2 'ros2' command found. Bag operations enabled.")
        else:
            self.get_logger().warn("ROS 2 'ros2' command not found. Bag operations will be disabled.")

        self.get_logger().info('Radar Point Cloud Analyzer node initialized with ROS2 bag support')

    def _check_ros2_availability(self) -> bool:
        """Check if the 'ros2' command is available in the system PATH."""
        import shutil
        return shutil.which("ros2") is not None

    def pcl_callback(self, msg: PointCloud2) -> None:
        """
        Process incoming radar PointCloud2 messages for real-time updates.
        
        This is the main data processing callback that updates visualization and
        collects data when active. It includes rate limiting to prevent excessive
        CPU usage during visualization.

        Args:
            msg: ROS 2 PointCloud2 message containing radar data.
        """
        # Update diagnostic information
        self.last_pcl_msg_time = self.get_clock().now()
        self.pcl_msg_count += 1
        
        # Get the timestamp from the message header
        try:
            timestamp_sec = msg.header.stamp.sec
            timestamp_nanosec = msg.header.stamp.nanosec
            current_timestamp = timestamp_sec + timestamp_nanosec * 1e-9
            
            # Add message to buffer for calibration lookup
            self.point_cloud_buffer.append((current_timestamp, msg))
            # Prune buffer
            now_ros = self.get_clock().now()
            now_sec = now_ros.nanoseconds / 1e9
            while self.point_cloud_buffer and (now_sec - self.point_cloud_buffer[0][0] > self.buffer_max_duration_sec):
                self.point_cloud_buffer.popleft()
            
        except Exception as e:
             self.get_logger().warn(f"Could not extract timestamp from PCL message: {e}")
             current_timestamp = time.time() # Fallback to current time
        
        # Force UI update during bag playback to ensure visualization
        if hasattr(self, 'is_playing') and self.is_playing:
            self.visible = True  # Ensure visibility during playback
        
        # Process point cloud data message - do this once to avoid duplicate processing
        try:
            points_list = list(pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
            ))
            
            if not points_list:
                self.get_logger().debug("Empty point cloud received")
                return
                
            # Convert to arrays just once
            x_array = np.array([p[1] for p in points_list], dtype=np.float32)
            y_array = np.array([p[0] for p in points_list], dtype=np.float32)
            z_array = np.array([p[2] for p in points_list], dtype=np.float32)
            intensities_array = np.array([p[3] for p in points_list], dtype=np.float32)
            
            # Store current data
            with self.data_lock:
                self.current_data['x'] = x_array
                self.current_data['y'] = y_array
                self.current_data['z'] = z_array
                self.current_data['intensities'] = intensities_array
                self.current_data['timestamp'] = current_timestamp # Store the timestamp
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")
            return
        
        # Handle bag playback point cloud data and update timeline
        if hasattr(self, 'is_playing') and self.is_playing:
            # Update playback progress for UI
            if hasattr(self, 'bag_duration') and self.bag_duration > 0 and hasattr(self, 'bag_start_time'):
                # Calculate elapsed time and position
                elapsed = time.time() - self.bag_start_time
                # Ensure position is between 0 and 1
                normalized_position = max(0.0, min(elapsed / self.bag_duration, 1.0))
                
                # Store the last update time and position to avoid unnecessary updates
                last_position_update_time = getattr(self, 'last_position_update_time', 0)
                last_position = getattr(self, 'last_position', -1.0)
                
                # Only update at most 10 times per second to avoid UI overload
                # Also update if position has changed significantly (1% or more)
                current_time = time.time()
                position_diff = abs(normalized_position - last_position)
                time_since_update = current_time - last_position_update_time
                
                if (time_since_update >= 0.1 or position_diff >= 0.01):
                    try:
                        # Make sure position is a valid float
                        if isinstance(normalized_position, (int, float)) and 0.0 <= normalized_position <= 1.0:
                            # Safely emit the signal through the signals object
                            try:
                                # Access the signals object directly to emit the signal
                                self.signals.update_playback_position_signal.emit(normalized_position)
                                self.get_logger().debug(
                                    f"Playback position: {normalized_position:.2f} "
                                    f"({elapsed:.2f}/{self.bag_duration:.2f}s)"
                                )
                            except Exception as signal_err:
                                self.get_logger().debug(
                                    f"Could not emit playback position signal: {str(signal_err)}"
                                )
                            
                            # Update the stored values
                            self.last_position_update_time = current_time
                            self.last_position = normalized_position
                    except Exception as e:
                        self.get_logger().debug(f"Error updating playback position: {str(e)}")
            
            # Log details every 20 frames to avoid excessive logging
            if self.pcl_msg_count % 20 == 0:
                self.get_logger().info(f"Received point cloud frame {self.pcl_msg_count} during bag playback")
                self.get_logger().info(f"  - Point cloud contains {len(x_array)} points")
        
        # Skip processing if not needed for visualization or collection
        if not self.visible and not self.collecting_data:
            return
            
        # Split processing paths for different operations based on mode
        try:
            if self.collecting_data:
                # Minimal processing for data collection (optimized path)
                self._process_for_data_collection(x_array, y_array, z_array, intensities_array)
            elif self.visible:
                # Check rate limiting for visualization updates only
                current_time = time.time()
                if current_time - self.last_update_time < self.update_interval:
                    return
                self.last_update_time = current_time
                
                # Full visualization processing
                self._process_for_visualization(x_array, y_array, z_array, intensities_array)
        except Exception as e:
            self.get_logger().error(f"Error in point cloud processing: {str(e)}")
            
    def _process_for_data_collection(self, x_array, y_array, z_array, intensities_array):
        """
        Optimized processing path for data collection during bag playback.
        
        This method only performs the minimum processing needed for data collection,
        skipping visualization-related operations to maximize performance.
        
        Args:
            x_array: X-coordinates of points
            y_array: Y-coordinates of points
            z_array: Z-coordinates of points
            intensities_array: Intensity values of points
        """
        with self.data_lock:
            # Process multi-frame point clouds if enabled
            # (optimized version will be handled in the multi_frame module)
            if self.params.enable_multi_frame:
                process_multi_frame_data(
                    self, x_array, y_array, z_array, intensities_array
                )
            
            # Process primary circle points only (optimized version in filter_points_in_circle)
            filter_points_in_circle(
                self, x_array, y_array, intensities_array
            )
            
            # Store data for collection
            if self.collection_start_time is not None:
                self.process_collected_data(z_array)
                elapsed_time = time.time() - self.collection_start_time
                if elapsed_time >= self.params.collection_duration:
                    self.stop_data_collection()
                    
    def _process_for_visualization(self, x_array, y_array, z_array, intensities_array):
        """
        Full processing path for visualization updates.
        
        This method performs all necessary operations for complete visualization
        of radar data including multi-frame processing, circle filtering, and
        heatmap generation.
        
        Args:
            x_array: X-coordinates of points
            y_array: Y-coordinates of points
            z_array: Z-coordinates of points
            intensities_array: Intensity values of points
        """
        with self.data_lock:
            # Process multi-frame point clouds if enabled
            if self.params.enable_multi_frame:
                process_multi_frame_data(
                    self, x_array, y_array, z_array, intensities_array
                )
            
            # Process all circle points for visualization
            filter_points_in_circle(
                self, x_array, y_array, intensities_array
            )

    def track_array_callback(self, msg: MarkerArray) -> None:
        """
        Process incoming radar track markers.
        
        This callback handles radar tracking information from the MarkerArray topic.
        Currently implemented as a placeholder for future functionality.

        Args:
            msg: ROS 2 MarkerArray message with track information.
        """
        # Placeholder for future implementation of track processing
        pass

    def occupancy_callback(self, msg: Bool) -> None:
        """
        Process incoming radar occupancy data.
        
        This callback handles occupancy information from the radar.
        Currently implemented as a placeholder for future functionality.

        Args:
            msg: Boolean indicating occupancy from the radar.
        """
        # Placeholder for future implementation of occupancy processing
        pass

    def timer_callback(self) -> None:
        """
        Regular timer callback for low-frequency tasks.
        
        This method is called periodically to handle diagnostics, heartbeats,
        monitoring, and low-priority tasks that don't need to run on every
        point cloud update.
        """
        # Update heartbeat for health monitoring
        self.last_heartbeat = self.get_clock().now()
        
        # Check bag playback process status
        if hasattr(self, 'rosbag_proc') and hasattr(self, 'is_playing') and hasattr(self, 'is_recording'):
            if (self.is_playing or self.is_recording) and self.rosbag_proc is not None:
                # Check if process is still running
                if self.rosbag_proc.poll() is not None:
                    # Process has terminated
                    exit_code = self.rosbag_proc.returncode
                    self.get_logger().info(f"ROS2 bag process has terminated with exit code: {exit_code}")
                    
                    # Reset recording/playing state
                    if self.is_recording:
                        self.is_recording = False
                        self.get_logger().info("Recording state reset after process termination")
                    
                    if self.is_playing:
                        self.is_playing = False
                        self.get_logger().info("Playback state reset after process termination")
                        # Notify UI that playback has ended
                        try:
                            self.signals.bag_playback_ended.emit()
                        except Exception as e:
                            self.get_logger().error(f"Failed to emit bag_playback_ended signal: {str(e)}")
                    
                    self.rosbag_proc = None
                    self.get_logger().info("ROS2 bag process has terminated")
        
        # Additional check for recording state - in case process was killed externally
        if hasattr(self, 'is_recording') and self.is_recording:
            if not hasattr(self, 'rosbag_proc') or self.rosbag_proc is None or self.rosbag_proc.poll() is not None:
                # Recording process is not running but state indicates recording
                self.get_logger().warn("Recording state inconsistency detected - resetting state")
                self.is_recording = False
                
                # Notify UI that recording has ended
                try:
                    self.signals.bag_playback_ended.emit()
                except Exception as e:
                    self.get_logger().error(f"Failed to emit bag_playback_ended signal: {str(e)}")
        
        # Existing timer callback code continues...
        # Monitor data collection progress if active
        if self.collecting_data and self.collection_start_time is not None:
            elapsed = time.time() - self.collection_start_time
            remaining = self.params.collection_duration - elapsed
            self.get_logger().info(
                f'Collecting data for {self.params.current_config}: '
                f'{int(elapsed)}s elapsed, {int(remaining)}s remaining'
            )
            
        # Monitor bag playback/recording and check if process has ended
        if hasattr(self, 'rosbag_proc') and hasattr(self, 'is_playing') and hasattr(self, 'is_recording'):
            if (self.is_playing or self.is_recording) and self.rosbag_proc is not None:
                # Check process status
                if self.rosbag_proc.poll() is not None:  # Process has ended
                    self.get_logger().info("ROS2 bag process has terminated")
                    # Perform hard reset of PCL data when playback ends
                    if self.is_playing:
                        self.hard_reset_pcl()
                        # Stop data collection if it was active
                        if self.collecting_data:
                            self.get_logger().info("Automatically stopping data collection as bag playback ended")
                            self.stop_data_collection()
                        # Reset visibility flag to stop processing
                        self.visible = False
                        # Reset message counters
                        self.pcl_msg_count = 0
                        self.last_pcl_msg_time = None
                        # Reset playback state
                        self.rosbag_proc = None
                        self.is_playing = False
                        self.is_recording = False
                        self.current_bag_path = None
                        self.bag_start_time = None
                        
                        # Emit signal to notify UI that bag playback has ended
                        try:
                            self.signals.bag_playback_ended.emit()
                            self.get_logger().info("Emitted bag_playback_ended signal to UI")
                        except Exception as e:
                            self.get_logger().error(f"Failed to emit bag_playback_ended signal: {str(e)}")
                            
                        self.get_logger().info("Analyzer stopped and reset after bag playback ended")
                
                # If bag is playing and not looping, check if it's nearing the end
                elif self.is_playing and hasattr(self, 'bag_looping') and not self.bag_looping:
                    if hasattr(self, 'bag_start_time') and hasattr(self, 'bag_duration') and self.bag_start_time is not None and self.bag_duration > 0:
                        current_time = time.time()
                        elapsed_playback = current_time - self.bag_start_time
                        
                        # If we're within 1 second of the end of the bag, prepare to finish up
                        if elapsed_playback >= (self.bag_duration - 1.0):
                            self.get_logger().info(f"Bag nearing end: {elapsed_playback:.1f}s of {self.bag_duration:.1f}s")
                            
                            # If we're at or past 95% of the bag duration, stop playback immediately
                            # Using 95% to ensure we stop before reaching the exact end which might be causing issues
                            if elapsed_playback >= (self.bag_duration * 0.95):
                                self.get_logger().info(f"Reached at least 95% of bag duration ({elapsed_playback:.1f}s of {self.bag_duration:.1f}s), stopping playback")
                                
                                # First, notify UI that playback has ended
                                try:
                                    self.signals.bag_playback_ended.emit()
                                    self.get_logger().info("Emitted bag_playback_ended signal to UI (end of bag)")
                                except Exception as e:
                                    self.get_logger().error(f"Failed to emit bag_playback_ended signal: {str(e)}")
                                
                                # Force terminate the bag playback process
                                if hasattr(self, 'rosbag_proc') and self.rosbag_proc is not None:
                                    try:
                                        import os
                                        import signal
                                        # Get process info
                                        pid = self.rosbag_proc.pid
                                        pgid = os.getpgid(pid)
                                        self.get_logger().info(f"Forcefully terminating ROS2 bag process {pid} (group {pgid})")
                                        # Send SIGTERM first
                                        os.killpg(pgid, signal.SIGTERM)
                                        # Reset state immediately
                                        self.is_playing = False
                                        self.visible = False
                                        # Wait very briefly
                                        time.sleep(0.5)
                                        # Send SIGKILL as backup if needed
                                        try:
                                            if self.rosbag_proc.poll() is None:  # Process still running
                                                os.killpg(pgid, signal.SIGKILL)
                                                self.get_logger().info(f"Sent SIGKILL to bag process {pid}")
                                        except Exception as kill_err:
                                            self.get_logger().error(f"Error sending SIGKILL: {str(kill_err)}")
                                    except Exception as term_err:
                                        self.get_logger().error(f"Error terminating bag process: {str(term_err)}")
                                
                                # Perform cleanup
                                self.hard_reset_pcl()
                                if self.collecting_data:
                                    self.stop_data_collection()
                                
                                # Additional reset of state variables
                                self.rosbag_proc = None
                                self.is_playing = False
                                self.current_bag_path = None
                                self.bag_start_time = None
                            else:
                                # If we've reached the bag duration, stop playback immediately
                                if elapsed_playback >= self.bag_duration:
                                    self.get_logger().info("Bag has reached its end, stopping playback")
                                    # Emit signal before stopping to ensure UI is notified
                                    try:
                                        self.signals.bag_playback_ended.emit()
                                        self.get_logger().info("Emitted bag_playback_ended signal to UI (end of bag)")
                                    except Exception as e:
                                        self.get_logger().error(f"Failed to emit bag_playback_ended signal: {str(e)}")
                                        
                                    stop_rosbag(self)
                                # Fallback: If we're way past the end and still playing, force stop
                                elif elapsed_playback > self.bag_duration + 2.0:
                                    self.get_logger().info("Bag should have ended by now, forcing stop")
                                    # Emit signal before stopping to ensure UI is notified
                                    try:
                                        self.signals.bag_playback_ended.emit()
                                        self.get_logger().info("Emitted bag_playback_ended signal to UI (timeout)")
                                    except Exception as e:
                                        self.get_logger().error(f"Failed to emit bag_playback_ended signal: {str(e)}")
                                        
                                    stop_rosbag(self)
                
                    # Monitor point cloud reception during playback
                    elif hasattr(self, 'last_pcl_msg_time') and self.last_pcl_msg_time is not None:
                        time_since_last_msg = (self.get_clock().now() - self.last_pcl_msg_time).nanoseconds / 1e9
                        if time_since_last_msg > 2.0:  # No messages for 2 seconds
                            self.get_logger().warn(
                                f"No point cloud messages received for {time_since_last_msg:.1f}s during bag playback"
                            )
                            # Try to read bag contents for diagnostics
                            if self.current_bag_path and os.path.exists(self.current_bag_path):
                                try:
                                    import subprocess
                                    self.get_logger().info(f"Checking topics in bag: {self.current_bag_path}")
                                    info_cmd = subprocess.run(
                                        ['ros2', 'bag', 'info', self.current_bag_path], 
                                        capture_output=True, text=True
                                    )
                                    self.get_logger().info(f"Bag info: {info_cmd.stdout}")
                                    
                                    # Print ROS topic list to see what's available
                                    self.get_logger().info("Checking active ROS topics:")
                                    topics_cmd = subprocess.run(
                                        ['ros2', 'topic', 'list'], 
                                        capture_output=True, text=True
                                    )
                                    self.get_logger().info(f"Active topics:\n{topics_cmd.stdout}")
                                    
                                    # Restart playback if needed
                                    if self.pcl_msg_count == 0 and self.is_playing:
                                        self.get_logger().warn("No point cloud messages received, restarting bag playback")
                                        self.stop_rosbag()
                                        # Wait a moment before restarting
                                        time.sleep(1.0)
                                        self.play_rosbag(self.current_bag_path)
                                except Exception as e:
                                    self.get_logger().error(f"Error during bag diagnostics: {str(e)}")

    def start_data_collection(self, config_name: str, target_distance: str, duration: int = 60) -> bool:
        """
        Start collecting radar data for a specified configuration and distance.
        
        This method initializes data structures for a new collection run and
        starts the collection process.

        Args:
            config_name: Name of the current radar configuration.
            target_distance: Target distance for the experiment (string to parse as float).
            duration: Collection duration in seconds.

        Returns:
            True if data collection started successfully, else False.
        """
        if self.collecting_data:
            self.get_logger().warn('Data collection already in progress')
            return False

        try:
            self.experiment_data.clear()

            self.params.current_config = config_name
            self.params.target_distance = float(target_distance)
            self.params.circle_distance = float(target_distance)
            self.params.collection_duration = duration

            # Keep primary sampling circle in sync
            if hasattr(self.params, 'circles') and len(self.params.circles) > 0:
                self.params.circles[0].distance = float(target_distance)
                
            # Update cached circle center
            self._cached_circle_center = np.array([0, self.params.circle_distance])

            self.collecting_data = True
            self.collection_start_time = time.time()
            
            # Ensure visualizations are updated during collection
            self.visible = True

            self.get_logger().info(
                f'Starting data collection for config {config_name} '
                f'at {target_distance}m for {duration}s'
            )
            return True
        except ValueError:
            self.get_logger().error(f"Invalid target distance: {target_distance}")
            return False
        except Exception as e:
            self.get_logger().error(f"Error starting data collection: {str(e)}")
            return False

    def stop_data_collection(self) -> None:
        """
        Stop collecting radar data and process/save results if any.
        
        This method finalizes the data collection, saves the collected data,
        and performs analysis on the results.
        """
        if not self.collecting_data:
            return

        self.collecting_data = False
        
        # Reset visualization visibility if not in playback mode
        if self.visible and not (hasattr(self, 'is_playing') and self.is_playing):
            self.visible = False

        if len(self.experiment_data.x_points) > 0:
            self.get_logger().info(
                f'Data collection complete. '
                f'Collected {len(self.experiment_data.x_points)} points'
            )
            
            # If multi-frame processing is enabled, finalize metrics
            if self.params.enable_multi_frame and self.multi_frame_metrics:
                self.experiment_data.multi_frame_metrics = self.multi_frame_metrics.copy()
                self.get_logger().info('Multi-frame metrics recorded')
            
            self.save_experiment_data()
            self.analyze_experiment_data()
        else:
            self.get_logger().warn('No data collected during the experiment')

    def save_experiment_data(self) -> None:
        """
        Save the collected experiment data to disk.
        
        This method saves points data, heatmap data, time-series data, and
        a visualization image to the experiment directory.
        """
        try:
            import pandas as pd
            
            data_dir = os.path.expanduser('~/radar_experiment_data')
            config_dir = os.path.join(data_dir, self.params.current_config)
            os.makedirs(config_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save points data
            data_file = os.path.join(
                config_dir,
                f"points_{int(self.params.target_distance)}m_{timestamp}.csv"
            )
            df = pd.DataFrame({
                'x': self.experiment_data.x_points,
                'y': self.experiment_data.y_points,
                'z': self.experiment_data.z_points,
                'intensity': self.experiment_data.intensities,
                'timestamp': self.experiment_data.timestamps,
                'target_distance': self.experiment_data.target_distances
            })
            df.to_csv(data_file, index=False)

            # Save time-series data
            if self.experiment_data.time_series_timestamps:
                ts_file = os.path.join(
                    config_dir,
                    f"time_series_{int(self.params.target_distance)}m_{timestamp}.csv"
                )
                # Prepare data dictionary for DataFrame, handling nested lists
                ts_data = {
                    'timestamp': self.experiment_data.time_series_timestamps
                }
                # Assuming 3 circles, create columns for each circle's count and avg_intensity
                num_circles = 3 # Should ideally get this dynamically if possible
                if self.experiment_data.circle_point_counts and len(self.experiment_data.circle_point_counts[0]) == num_circles:
                    for i in range(num_circles):
                        # Get label for column naming (fallback to index)
                        label = self.params.circles[i].label if i < len(self.params.circles) else f'circle{i}'
                        label = label.lower().replace(" ", "_") # Make label CSV-friendly
                        
                        ts_data[f'{label}_points'] = [counts[i] for counts in self.experiment_data.circle_point_counts]
                        ts_data[f'{label}_avg_intensity'] = [intensities[i] for intensities in self.experiment_data.circle_avg_intensities]
                else:
                     self.get_logger().warn("Time-series data format mismatch or empty. Skipping detailed circle columns.")
                     # Fallback to old single columns if data structure is unexpected
                     ts_data['circle_points'] = [item[0] if isinstance(item, list) and item else 0 for item in self.experiment_data.circle_point_counts]
                     ts_data['circle_avg_intensity'] = [item[0] if isinstance(item, list) and item else 0.0 for item in self.experiment_data.circle_avg_intensities]

                df_ts = pd.DataFrame(ts_data)
                df_ts.to_csv(ts_file, index=False)
                self.get_logger().info(f'Saved time series data to {ts_file}')
                
            # Save multi-frame metrics if available
            if self.experiment_data.multi_frame_metrics:
                mf_file = os.path.join(
                    config_dir,
                    f"multi_frame_{int(self.params.target_distance)}m_{timestamp}.json"
                )
                with open(mf_file, 'w') as f:
                    import json
                    json.dump(self.experiment_data.multi_frame_metrics, f, indent=2)
                self.get_logger().info(f'Saved multi-frame metrics to {mf_file}')

            save_visualization(self, config_dir, timestamp)
        except Exception as e:
            self.get_logger().error(f"Error saving experiment data: {str(e)}")

    def analyze_experiment_data(self) -> None:
        """
        Analyze the collected experiment data.
        
        This method computes statistics from the collected data, including
        distance bands analysis and circle statistics. Results are stored
        in self.config_results for later reporting.
        """
        if len(self.experiment_data.x_points) == 0:
            self.get_logger().warn("No data to analyze")
            return

        try:
            # Convert to numpy arrays for faster computation
            x_array = np.array(self.experiment_data.x_points, dtype=np.float32)
            y_array = np.array(self.experiment_data.y_points, dtype=np.float32)
            intensities_array = np.array(self.experiment_data.intensities, dtype=np.float32)
            timestamps_array = np.array(self.experiment_data.timestamps, dtype=np.float32)
            
            # Get the frame count for analysis - use all frames by default for more accurate total counts
            frames_to_analyze = getattr(self.params, 'analysis_frame_count', None)  # Use configured value if available
            
            # If no specific configuration, use all available frames
            if frames_to_analyze is None:
                if len(timestamps_array) > 0:
                    unique_timestamps = np.unique(timestamps_array)
                    frames_to_analyze = len(unique_timestamps)
                    self.get_logger().info(f"Using all {frames_to_analyze} frames for distance band analysis")
                else:
                    frames_to_analyze = 10  # Default fallback if no timestamp data
            
            # Optional: Allow limiting to recent frames via configuration
            use_recent_frames_only = getattr(self.params, 'use_recent_frames_only', False)
            
            # Limit analysis to specific frames if requested
            if use_recent_frames_only and len(timestamps_array) > 0:
                # Get unique timestamps
                unique_timestamps = np.unique(timestamps_array)
                
                # Define recent frames to use (all or limited)
                max_frames_to_use = min(frames_to_analyze, 10)  # Don't use more than 10 for "recent" frames
                
                # If we have more frames than max_frames_to_use, use only the most recent ones
                if len(unique_timestamps) > max_frames_to_use:
                    # Sort and get the most recent timestamps
                    recent_timestamps = np.sort(unique_timestamps)[-max_frames_to_use:]
                    
                    # Create a mask for the most recent frames
                    recent_mask = np.isin(timestamps_array, recent_timestamps)
                    
                    # Apply the mask to all arrays
                    x_array = x_array[recent_mask]
                    y_array = y_array[recent_mask]
                    intensities_array = intensities_array[recent_mask]
                    
                    self.get_logger().info(f"Limited analysis to {max_frames_to_use} most recent frames ({len(x_array)} points)")
            else:
                self.get_logger().info(f"Analyzing all available data: {len(x_array)} points from {frames_to_analyze} frames")
            
            # Calculate distances from origin using all data
            # For radar, forward might be y-axis positive; we may need to consider direction
            
            # Standard Euclidean distance (radial from origin)
            distances_euclidean = np.sqrt(x_array ** 2 + y_array ** 2)
            
            # Alternative: distance along primary axis (Y-axis for radar typically points forward)
            # Use absolute values to ensure we're measuring distance regardless of direction
            distances_forward = np.abs(y_array)
            
            # Determine which distance calculation to use based on parameters
            use_directional = getattr(self.params, 'use_directional_distance', False)
            
            if use_directional:
                # Use directional distance (along forward axis)
                distances = distances_forward
                self.get_logger().info("Using directional (Y-axis) distance calculation")
            else:
                # Use standard Euclidean distance from origin
                distances = distances_euclidean
                self.get_logger().info("Using radial/Euclidean distance calculation")
            
            # Create distance bins
            bins = np.arange(0, self.params.max_range + self.params.circle_interval, self.params.circle_interval)
            counts, _ = np.histogram(distances, bins=bins)

            distance_bands = {}
            for i in range(len(counts)):
                band_key = f"{bins[i]}-{bins[i+1]}m"
                band_mask = (distances >= bins[i]) & (distances < bins[i+1])
                band_intensities = intensities_array[band_mask]
                band_count = float(np.sum(band_mask))  # More reliable count calculation
                avg_intensity = float(np.mean(band_intensities)) if band_intensities.size > 0 else 0.0
                distance_bands[band_key] = {'count': band_count, 'avg_intensity': avg_intensity}

            if self.params.current_config not in self.config_results:
                self.config_results[self.params.current_config] = {}

            # Find the band containing the target distance
            target_band_key = None
            target_distance = self.params.target_distance
            
            # First try exact matching using the target distance
            for band_key in distance_bands.keys():
                band_start, band_end = band_key.split('-')
                band_start = float(band_start)
                band_end = float(band_end.replace('m', ''))
                
                if band_start <= target_distance < band_end:
                    target_band_key = band_key
                    self.get_logger().info(f"Target band for {target_distance}m identified as {target_band_key}")
                    break
            
            # Fallback to the first band if target band not found
            if target_band_key is None:
                target_band_key = list(distance_bands.keys())[0] if distance_bands else "0-0m"
                self.get_logger().warn(f"Could not find exact band for {target_distance}m, using {target_band_key}")

            # Double-check the target band points with direct calculation
            target_band_start, target_band_end = target_band_key.split('-')
            target_band_start = float(target_band_start)
            target_band_end = float(target_band_end.replace('m', ''))
            
            # Create mask for target band points and count them directly
            target_band_mask = (distances >= target_band_start) & (distances < target_band_end)
            target_band_points = float(np.sum(target_band_mask))
            
            # Log detailed diagnostic information
            self.get_logger().info(f"Points in distance range {target_band_start}-{target_band_end}m: {target_band_points}")
            self.get_logger().info(f"Total points analyzed: {len(distances)}")
            
            # Special handling for specific bands for easier diagnostics
            for band_key, band_data in distance_bands.items():
                band_start, band_end = band_key.split('-')
                band_start = float(band_start)
                band_end = float(band_end.replace('m', ''))
                band_mask = (distances >= band_start) & (distances < band_end)
                actual_count = float(np.sum(band_mask))
                
                if abs(actual_count - band_data['count']) > 0.01:
                    self.get_logger().warn(f"Count mismatch in band {band_key}: stored={band_data['count']}, actual={actual_count}")
                    # Update to the correct count
                    distance_bands[band_key]['count'] = actual_count

            # Use all points for calculations
            circle_counts = len(x_array)
            circle_avg_intensity = float(np.mean(intensities_array)) if intensities_array.size > 0 else 0.0

            # Store results in config_results
            results_dict = {
                'total_points': circle_counts,
                'distance_bands': distance_bands,
                'target_band': target_band_key,
                'target_band_points': target_band_points,  # Use directly calculated value
                'avg_intensity': circle_avg_intensity,
                'circle_points': circle_counts,
                'circle_avg_intensity': circle_avg_intensity,
                'frames_analyzed': frames_to_analyze  # Store how many frames were used in the analysis
            }
            
            # Add multi-frame metrics if available
            if self.experiment_data.multi_frame_metrics:
                results_dict['multi_frame_metrics'] = self.experiment_data.multi_frame_metrics.copy()
                
                # Copy ROI-specific metrics to the top level for easier access in reports
                if 'roi_combined_point_count' in self.experiment_data.multi_frame_metrics:
                    results_dict['roi_combined_point_count'] = self.experiment_data.multi_frame_metrics['roi_combined_point_count']
                    results_dict['roi_avg_single_frame_count'] = self.experiment_data.multi_frame_metrics.get('roi_avg_single_frame_count', 0)
                    results_dict['roi_spatial_density'] = self.experiment_data.multi_frame_metrics.get('roi_spatial_density', 0)
                    results_dict['roi_snr_db'] = self.experiment_data.multi_frame_metrics.get('roi_snr_db', 0)
                
                # Copy secondary ROI metrics (roi2, roi3) to the top level for easier access in reports
                for i in range(1, len(self.params.circles)):
                    if self.params.circles[i].enabled:
                        prefix = f'roi{i+1}'
                        if f'{prefix}_combined_point_count' in self.experiment_data.multi_frame_metrics:
                            self.get_logger().info(f"Adding {prefix} metrics to top level for reports")
                            
                            # Copy the metrics to the top level
                            results_dict[f'{prefix}_combined_point_count'] = self.experiment_data.multi_frame_metrics[f'{prefix}_combined_point_count']
                            results_dict[f'{prefix}_avg_single_frame_count'] = self.experiment_data.multi_frame_metrics.get(f'{prefix}_avg_single_frame_count', 0)
                            results_dict[f'{prefix}_spatial_density'] = self.experiment_data.multi_frame_metrics.get(f'{prefix}_spatial_density', 0)
                            results_dict[f'{prefix}_snr_db'] = self.experiment_data.multi_frame_metrics.get(f'{prefix}_snr_db', 0)
                            results_dict[f'{prefix}_combined_min_intensity'] = self.experiment_data.multi_frame_metrics.get(f'{prefix}_combined_min_intensity', 0)
                            results_dict[f'{prefix}_combined_max_intensity'] = self.experiment_data.multi_frame_metrics.get(f'{prefix}_combined_max_intensity', 0)
                            results_dict[f'{prefix}_combined_avg_intensity'] = self.experiment_data.multi_frame_metrics.get(f'{prefix}_combined_avg_intensity', 0)
                        else:
                            # Skip the debug warnings about missing metrics
                            pass
                
                # Also include outside ROI metrics at top level
                if 'outside_roi_combined_point_count' in self.experiment_data.multi_frame_metrics:
                    results_dict['outside_roi_combined_point_count'] = self.experiment_data.multi_frame_metrics['outside_roi_combined_point_count']
                    results_dict['outside_roi_avg_single_frame_count'] = self.experiment_data.multi_frame_metrics.get('outside_roi_avg_single_frame_count', 0)
                    results_dict['outside_roi_spatial_density'] = self.experiment_data.multi_frame_metrics.get('outside_roi_spatial_density', 0)
                    results_dict['outside_roi_snr_db'] = self.experiment_data.multi_frame_metrics.get('outside_roi_snr_db', 0)
                
                self.get_logger().info(f"Added multi-frame metrics with {len(self.experiment_data.multi_frame_metrics)} keys")
                
                # Log available ROI metrics keys for debugging
                roi_keys = [k for k in self.experiment_data.multi_frame_metrics.keys() if k.startswith('roi')]
                if roi_keys:
                    self.get_logger().info(f"ROI metric keys: {', '.join(roi_keys)}")
            
            # Add metadata if available
            if hasattr(self.experiment_data, 'metadata') and self.experiment_data.metadata:
                # Make a deep copy to avoid overwriting 
                results_dict['metadata'] = dict(self.experiment_data.metadata)
                
                # Use more accurate metadata info for target band if available
                if 'target_band' in self.experiment_data.metadata:
                    results_dict['target_band'] = self.experiment_data.metadata['target_band']
                if 'target_band_count' in self.experiment_data.metadata:
                    results_dict['target_band_points'] = self.experiment_data.metadata['target_band_count']
                
                self.get_logger().info(f'Using detailed metadata with {len(self.experiment_data.metadata)} keys')
                # Log keys found in metadata for debugging
                self.get_logger().info(f'Metadata keys: {", ".join(self.experiment_data.metadata.keys())}')
            
            self.config_results[self.params.current_config][int(self.params.target_distance)] = results_dict

            self.get_logger().info(
                f'Analyzed data for {self.params.current_config} at {self.params.target_distance}m'
            )
            self.get_logger().info(f'Total points in sampling circle: {circle_counts}')
            self.get_logger().info(f'Average intensity in sampling circle: {circle_avg_intensity:.2f}')
            
            # Log target band information if available
            target_band_points = results_dict.get('target_band_points', 0)
            if target_band_points > 0:
                self.get_logger().info(
                    f'Points in target band {target_band_key}: {target_band_points} '
                    f'({target_band_points/circle_counts*100:.1f}% of total)'
                )
                
        except Exception as e:
            self.get_logger().error(f"Error analyzing experiment data: {str(e)}")

    def process_collected_data(self, z_array: np.ndarray) -> None:
        """
        Collect and record data from points in all enabled sampling circles for each new frame.
        
        This method adds points within all enabled sampling circles to the experiment data
        and updates the heatmap with these points. Points are carefully categorized by distance
        bands for more accurate analysis.

        Args:
            z_array: Z-coordinates of the current point cloud.
        """
        try:
            timestamp = (
                self.get_clock().now().to_msg().sec
                + self.get_clock().now().to_msg().nanosec * 1e-9
            )
            
            # --- Process data for ALL enabled circles --- 
            counts_per_circle = []
            avg_intensities_per_circle = []
            all_x = []
            all_y = []
            all_z = []
            all_intensities = []
            all_timestamps = []
            all_target_distances = []

            for i, circle_param in enumerate(self.params.circles):
                if not circle_param.enabled:
                    # Append default values for disabled circles
                    counts_per_circle.append(0)
                    avg_intensities_per_circle.append(0.0)
                    continue

                # Construct keys for accessing circle-specific data in self.current_data
                circle_key_prefix = f'circle{i+1}' if i > 0 else 'circle' # 'circle' for index 0, 'circle2' for 1, 'circle3' for 2
                x_key = f'{circle_key_prefix}_x' if i > 0 else 'circle_x'
                y_key = f'{circle_key_prefix}_y' if i > 0 else 'circle_y'
                intensity_key = f'{circle_key_prefix}_intensities' if i > 0 else 'circle_intensities'
                index_key = f'{circle_key_prefix}_indices' if i > 0 else 'circle_indices'

                # Get data for the current circle
                x_data = self.current_data.get(x_key, np.array([], dtype=np.float32))
                y_data = self.current_data.get(y_key, np.array([], dtype=np.float32))
                intensities_data = self.current_data.get(intensity_key, np.array([], dtype=np.float32))
                indices = self.current_data.get(index_key, np.array([], dtype=np.int32))
                
                current_circle_count = len(indices) if indices is not None and indices.size > 0 else 0
                current_avg_intensity = float(np.mean(intensities_data)) if current_circle_count > 0 else 0.0
                
                counts_per_circle.append(current_circle_count)
                avg_intensities_per_circle.append(current_avg_intensity)
                
                # Collect point data if circle has points
                if current_circle_count > 0 and len(indices) <= len(z_array):
                    all_x.extend(x_data.tolist())
                    all_y.extend(y_data.tolist())
                    all_z.extend(z_array[indices].tolist())
                    all_intensities.extend(intensities_data.tolist())
                    all_timestamps.extend([timestamp] * current_circle_count)
                    all_target_distances.extend([self.params.target_distance] * current_circle_count)
                    
                    self.get_logger().debug(
                        f"Circle {i} ('{circle_param.label}'): Added {current_circle_count} points. "
                        f"Avg Intensity: {current_avg_intensity:.2f}"
                    )
                else:
                    self.get_logger().debug(f"Circle {i} ('{circle_param.label}'): No points found or index mismatch.")

            # Append the collected lists for this timestamp to ExperimentData
            if any(count > 0 for count in counts_per_circle):
                self.experiment_data.time_series_timestamps.append(timestamp)
                self.experiment_data.circle_point_counts.append(counts_per_circle)
                self.experiment_data.circle_avg_intensities.append(avg_intensities_per_circle)
                
                # Extend point data
                self.experiment_data.x_points.extend(all_x)
                self.experiment_data.y_points.extend(all_y)
                self.experiment_data.z_points.extend(all_z)
                self.experiment_data.intensities.extend(all_intensities)
                self.experiment_data.timestamps.extend(all_timestamps)
                self.experiment_data.target_distances.extend(all_target_distances)
                
                # Format the avg intensities list for logging
                avg_intensities_str = ", ".join([f'{v:.2f}' for v in avg_intensities_per_circle])
                self.get_logger().debug(
                    f"Appended time series data for timestamp {timestamp}: "
                    f"Counts={counts_per_circle}, AvgIntensities=[{avg_intensities_str}]"
                )
            else:
                # Optionally, still append timestamp even if no points found in any circle?
                # self.experiment_data.time_series_timestamps.append(timestamp)
                # self.experiment_data.circle_point_counts.append([0, 0, 0])
                # self.experiment_data.circle_avg_intensities.append([0.0, 0.0, 0.0])
                self.get_logger().debug(f"No points found in any enabled circle for timestamp {timestamp}. Skipping append.")

        except Exception as e:
            self.get_logger().error(f"Error in process_collected_data: {str(e)}")
            
    def hard_reset_pcl(self) -> None:
        """
        Perform a hard reset of all PCL-related data structures.
        
        This method clears all point cloud data, heatmaps, and related visualizations
        to ensure a clean state for the next playback.
        """
        try:
            with self.data_lock:
                # Reset current data arrays for main point cloud
                self.current_data = {
                    'x': np.array([], dtype=np.float32),
                    'y': np.array([], dtype=np.float32),
                    'z': np.array([], dtype=np.float32),
                    'intensities': np.array([], dtype=np.float32),
                    'circle_x': np.array([], dtype=np.float32),
                    'circle_y': np.array([], dtype=np.float32),
                    'circle_intensities': np.array([], dtype=np.float32),
                    'circle_indices': np.array([], dtype=np.int32)
                }
                
                # Initialize data arrays for additional circles
                for i in range(1, len(self.params.circles)):
                    circle_key = f'circle{i+1}'
                    self.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                    self.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                    self.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                    self.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
                
                # Reset heatmap data
                grid_size = calculate_heatmap_size(self.params)
                self.heatmap_data = np.zeros(grid_size, dtype=np.float32)
                self.live_heatmap_data = np.zeros(grid_size, dtype=np.float32)
                
                # Reset frame buffer if multi-frame processing is enabled
                if self.params.enable_multi_frame:
                    self.frame_buffer = []
                    self.frame_count = 0  # Reset frame count for memory management
                    
                    # Reset combined frame data too
                    self.combined_frame = {
                        'x': np.array([], dtype=np.float32),
                        'y': np.array([], dtype=np.float32),
                        'z': np.array([], dtype=np.float32),
                        'intensities': np.array([], dtype=np.float32)
                    }
                    self.multi_frame_metrics = {}
                
                # Reset visualization components if they exist
                if self.viz_components['scatter'] is not None:
                    self.viz_components['scatter'].set_offsets(np.empty((0, 2)))
                if self.viz_components['circle_scatter'] is not None:
                    self.viz_components['circle_scatter'].set_offsets(np.empty((0, 2)))
                
                # IMPORTANT: Also clear experiment_data to avoid keeping stale data
                if hasattr(self, 'experiment_data'):
                    self.get_logger().info("Clearing experiment_data during hard reset")
                    self.experiment_data.clear()
                
                # Emit signal to notify UI that data has been reset
                try:
                    self.signals.data_reset_signal.emit()
                    self.get_logger().debug("Emitted data reset signal to UI")
                except Exception as e:
                    self.get_logger().debug(f"Failed to emit data reset signal: {str(e)}")
                
                self.get_logger().info("Performed hard reset of PCL data structures")
                
        except Exception as e:
            self.get_logger().error(f"Error during PCL hard reset: {str(e)}")
            
    def play_rosbag(self, bag_path: str, loop: bool = False) -> bool:
        """
        Play a ROS2 bag file.
        
        Wrapper around the play_rosbag function that maintains
        class state for playback operations.
        
        Args:
            bag_path: Path to the ROS2 bag file to play
            loop: Whether to loop the playback when it ends (defaults to False)
            
        Returns:
            Success status of the playback operation
        """
        from radar_analyzer.utils.ros_bag_handler import play_rosbag as play_rosbag_func
        try:
            # Clear experiment data when starting a new bag playback
            if hasattr(self, 'experiment_data') and not self.collecting_data:
                self.get_logger().info("Clearing experiment data before starting bag playback")
                with self.data_lock:
                    self.experiment_data.clear()
            
            play_rosbag_func(self, bag_path, loop)
            return True
        except Exception as e:
            self.get_logger().error(f"Error in play_rosbag: {str(e)}")
            return False

    def record_rosbag(self, output_path: str, topics: List[str] = None, duration_minutes: int = 0) -> bool:
        """
        Record a ROS2 bag file.
        
        Wrapper around the record_rosbag function that maintains
        class state for recording operations.
        
        Args:
            output_path: Path to save the ROS2 bag file
            topics: List of topics to record, if None, all topics are recorded
            duration_minutes: Duration in minutes to record (0 = unlimited)
            
        Returns:
            Success status of the recording operation
        """
        from radar_analyzer.utils.ros_bag_handler import record_rosbag as record_rosbag_func
        try:
            record_rosbag_func(self, output_path, topics if topics else [], duration_minutes)
            return True
        except Exception as e:
            self.get_logger().error(f"Error in record_rosbag: {str(e)}")
            return False

    def stop_rosbag(self) -> bool:
        """
        Stop an active ROS2 bag playback or recording.
        
        Wrapper around the stop_rosbag function that maintains
        class state for bag operations. Ensures all data is properly
        reset when stopping.
        
        Returns:
            Success status of the stop operation
        """
        from radar_analyzer.utils.ros_bag_handler import stop_rosbag as stop_rosbag_func
        try:
            was_playing = self.is_playing
            stop_rosbag_func(self)
            
            # If we were playing (not just recording), perform hard reset
            if was_playing:
                self.get_logger().info("Performing hard reset after stopping bag playback")
                self.hard_reset_pcl()
                
                # If data collection was active, stop it
                if self.collecting_data:
                    self.get_logger().info("Stopping data collection after bag playback ended")
                    self.stop_data_collection()
            
            return True
        except Exception as e:
            self.get_logger().error(f"Error in stop_rosbag: {str(e)}")
            return False

    def seek_rosbag(self, position: float) -> bool:
        """
        Seek to a specific position in a ROS2 bag playback.
        
        Wrapper around the seek_rosbag function that maintains
        class state for playback operations.
        
        Args:
            position: Normalized position in the bag (0.0-1.0)
            
        Returns:
            Success status of the seek operation
        """
        from radar_analyzer.utils.ros_bag_handler import seek_rosbag as seek_rosbag_func
        try:
            seek_rosbag_func(self, position)
            return True
        except Exception as e:
            self.get_logger().error(f"Error in seek_rosbag: {str(e)}")
            return False

    def compute_heatmap_metrics(self) -> Dict[str, float]:
        """
        Compute metrics based on the current live heatmap data.

        Calculates statistics like max/average intensity, SNR, and coverage.
        These metrics provide insights into the overall signal strength and distribution
        in the radar's field of view.
        
        Returns:
            Dictionary containing computed heatmap metrics.
        """
        # Removed heatmap metrics computation
        self.get_logger().info("Heatmap metrics calculation removed.")
        return {
            'max_intensity': 0.0,
            'avg_intensity': 0.0,
            'snr_dB': 0.0,
            'active_cells': 0.0,
            'total_cells': 1.0,
            'coverage_percentage': 0.0
        }

    def update_circle_position(self, distance: float) -> None:
        """
        Update the vertical position of the sampling circle in both scatter and heatmap.
        
        This is a wrapper around the update_circle_position function from the
        visualizer module that maintains proper error handling.
        
        Args:
            distance: New vertical position (distance) for the circle center.
        """
        from radar_analyzer.visualization.visualizer import update_circle_position as update_position_func
        try:
            update_position_func(self, distance)
        except Exception as e:
            self.get_logger().error(f"Error updating circle position: {str(e)}")
            
    def update_circle_radius(self, radius: float) -> None:
        """
        Update the radius of the sampling circle.
        
        Args:
            radius: New circle radius in meters.
        """
        self.params.circle_radius = radius
        # Update visualization component
        update_circle_radius(self)
        self.get_logger().info(f'Sampling circle radius updated to {radius}m')

    # --- Start Methods for Calibration Point Processing (Paper Algorithm 1) --- 
    def _get_points_from_msgs_in_window(self, center_timestamp: float) -> pd.DataFrame | None:
        """
        Retrieves and parses points from buffered PointCloud2 messages
        within the defined time window. Calculates range.
        """
        start_time = center_timestamp - self.time_window_sec / 2
        end_time = center_timestamp + self.time_window_sec / 2
        points_list = []
        relevant_msgs = 0
        required_fields = {'x', 'y', 'z'}

        # Iterate through a copy of the deque for safety
        for ts, msg in list(self.point_cloud_buffer):
            try:
                if start_time <= ts <= end_time:
                    relevant_msgs += 1
                    # Determine required fields based on config
                    fields_to_read = list(required_fields) # Start with x,y,z
                    # Check if velocity field *actually* exists in this specific message
                    msg_field_names = {field.name for field in msg.fields}
                    if not required_fields.issubset(msg_field_names):
                        self.get_logger().warn(f"Skipping msg at {ts:.3f}: missing x, y, or z.", throttle_duration_sec=10)
                        continue # Skip message if core fields missing
                        
                    has_velocity = self.use_velocity_filter and self.velocity_field in msg_field_names

                    if has_velocity:
                        fields_to_read.append(self.velocity_field)

                    # Use standard sensor_msgs_py.point_cloud2 to read points
                    try:
                        # Setting skip_nans=True is important
                        for point_struct in pc2.read_points(msg, field_names=fields_to_read, skip_nans=True):
                            # point_struct is now a tuple/list of values based on field_names order
                            # Convert to a dictionary for easier DataFrame creation
                            point_data = {'timestamp': ts}
                            # Assuming standard field order (x, y, z, optional_velocity)
                            point_data['x'] = float(point_struct[0])
                            point_data['y'] = float(point_struct[1])
                            point_data['z'] = float(point_struct[2])
                            if has_velocity:
                                point_data['velocity'] = float(point_struct[3]) # Assuming velocity is 4th field
                            else:
                                point_data['velocity'] = 0.0 # Assign 0 if not present or not used

                            # Calculate range
                            point_data['range'] = np.sqrt(point_data['x']**2 + point_data['y']**2 + point_data['z']**2)
                            points_list.append(point_data)
                    except Exception as e:
                        self.get_logger().error(f"Error parsing PointCloud2 message from timestamp {ts}: {e}")
                        # Continue to next message
            except Exception as e:
                self.get_logger().error(f"Outer error during PCL message processing loop for timestamp {ts}: {e}")

        if not points_list:
            self.get_logger().warn(f"No points extracted from {relevant_msgs} messages in window [{start_time:.3f}, {end_time:.3f}]")
            return None

        self.get_logger().info(f"Extracted {len(points_list)} points from {relevant_msgs} messages in window.")
        return pd.DataFrame(points_list)

    # --- This is the core method called by CalibrationView --- 
    def find_sync_corner_reflector_position(self, target_timestamp: float):
        """
        Finds the filtered and averaged 3D position of a static Corner Reflector (CR)
        based on radar PointCloud2 data around a specific timestamp.

        Implements Algorithm 1, lines 5-13 from Cheng et al., arXiv:2307.15264.
        """
        self.get_logger().info(f"Requesting CR position near image timestamp {target_timestamp:.3f}")
        processing_start_time = time.time()

        # --- 1. Data Aggregation & Parsing --- 
        # Get DataFrame of points within the time window, including calculated range
        points_df = self._get_points_from_msgs_in_window(target_timestamp)

        if points_df is None or points_df.empty:
            self.get_logger().warn("No points found in the specified time window.")
            return None
        self.get_logger().debug(f"Initial points count: {len(points_df)}")

        # --- 2. Initial Filtering (Static & Range) --- 
        # Paper implies velocity=0 check first (line 5 hint, line 9 context)
        if self.use_velocity_filter:
            # Make sure the velocity column exists (added in _get_points...)
             if 'velocity' in points_df.columns:
                 static_mask = np.abs(points_df['velocity']) < 1e-6 # Threshold for static
                 self.get_logger().debug(f"Points before velocity filter: {len(points_df)}")
                 points_df = points_df[static_mask].copy()
                 self.get_logger().debug(f"Points after velocity filter: {len(points_df)}")
             else:
                  self.get_logger().warn("Velocity filtering enabled but 'velocity' column missing from extracted points.")

        # Filter by range
        range_mask = points_df['range'] < self.range_threshold_m
        self.get_logger().debug(f"Points before range filter: {len(points_df)}")
        points_df = points_df[range_mask].copy()
        self.get_logger().debug(f"Points after range filter (<{self.range_threshold_m}m): {len(points_df)}")


        if points_df.empty:
            self.get_logger().warn("No points remaining after static/range filtering.")
            return None

        # --- 3. Outlier Removal (Z-Score) (Algorithm 1, line 9-10) --- 
        coords = points_df[['x', 'y', 'z']]
        if len(coords) < 2: # Need at least 2 points to calculate Z-score reliably
             self.get_logger().warn(f"Too few points ({len(coords)}) to perform Z-score filtering.")
             # Depending on requirements, either return None or proceed without Z-score
             inliers = points_df
        else:
            try:
                # Ensure coords are numeric before zscore
                coords_numeric = coords.apply(pd.to_numeric, errors='coerce').dropna()
                if coords_numeric.empty:
                    self.get_logger().warn("No numeric coordinate data left for Z-score.")
                    inliers = points_df # Fallback
                else:
                    z_scores = np.abs(stats.zscore(coords_numeric))
                    # Filter based on threshold for all axes simultaneously
                    all_axes_inliers_mask = np.all(z_scores < self.z_score_thresh, axis=1)
                    # Apply mask back to the original DataFrame's index that corresponds to numeric data
                    inliers = points_df.loc[coords_numeric.index[all_axes_inliers_mask]].copy()
                    self.get_logger().debug(f"Points after Z-score filter (thresh={self.z_score_thresh}): {len(inliers)}")
            except Exception as e:
                self.get_logger().error(f"Error during Z-score calculation: {e}. Skipping Z-score filter.")
                inliers = points_df # Fallback to using points before Z-score

        if inliers.empty:
            self.get_logger().warn("No points remaining after Z-score filtering.")
            return None
        num_inliers = len(inliers)

        # --- 4. Check Minimum Inliers --- 
        if num_inliers < self.min_inlier_points:
            self.get_logger().warn(f"Insufficient inliers ({num_inliers}) after filtering (minimum required: {self.min_inlier_points}). Cannot calculate reliable mean.")
            return None

        # --- 5. Averaging Inliers (Algorithm 1, line 11) --- 
        mean_position = inliers[['x', 'y', 'z']].mean().to_list() # Calculate mean

        processing_time_ms = (time.time() - processing_start_time) * 1000
        self.get_logger().info(f"Calculated mean CR position: {tuple(mean_position)} from {num_inliers} inliers. Processing time: {processing_time_ms:.1f} ms")
        return tuple(mean_position)

    def get_latest_radar_points(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Safely returns the latest processed radar point coordinates."""
        with self.data_lock:
            if ('x' in self.current_data and self.current_data['x'].size > 0 and
                'y' in self.current_data and self.current_data['y'].size > 0 and
                'z' in self.current_data and self.current_data['z'].size > 0):
                # Return copies to avoid external modification issues
                return (
                    self.current_data['x'].copy(),
                    self.current_data['y'].copy(),
                    self.current_data['z'].copy()
                )
            else:
                return None # No valid data available
    # --- End Calibration Point Processing Methods ---

    # --- Placeholder for Checkerboard-based Extrinsic Calibration ---
    def find_sync_checkerboard_pose(self, target_timestamp: float, time_tolerance: float = 0.1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        (Placeholder) Estimate the 3D pose (rvec, tvec) of the checkerboard 
        in the radar coordinate system, synchronized with the camera frame.

        Args:
            target_timestamp: Timestamp of the camera frame.
            time_tolerance: Max allowed time difference.

        Returns:
            Tuple (rvec, tvec) representing the board's pose in radar frame, or None.
        """
        self.get_logger().warn(f"Checkerboard pose detection not yet implemented. Called with timestamp {target_timestamp:.3f}")
        return None
    # --- End Placeholder ---