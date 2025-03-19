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
    calculate_heatmap_size,
    update_heatmap_vectorized,
    update_live_heatmap_vectorized,
    apply_live_heatmap_decay,
    compute_heatmap_metrics
)
from radar_analyzer.visualization.visualizer import (
    setup_visualization,
    setup_heatmap_visualization,
    update_plot,
    update_heatmap_display,
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
        self.combined_frame = {
            'x': np.array([], dtype=np.float32),
            'y': np.array([], dtype=np.float32),
            'z': np.array([], dtype=np.float32),
            'intensities': np.array([], dtype=np.float32)
        }
        self.multi_frame_metrics = {}

        # Create a more robust QoS profile for better bag playback compatibility
        self.reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Use the reliable QoS profile for all subscriptions
        self.pcl_subscription = self.create_subscription(
            PointCloud2, '/ti_mmwave/radar_scan_pcl', self.pcl_callback, self.reliable_qos
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

        # Heatmap data
        grid_size = calculate_heatmap_size(self.params)
        self.heatmap_data = np.zeros(grid_size, dtype=np.float32)
        self.live_heatmap_data = np.zeros(grid_size, dtype=np.float32)
        self.live_heatmap_decay_factor = 0.98

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

        self.get_logger().info('Radar Point Cloud Analyzer node initialized with ROS2 bag support')

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
        
        # Force UI update during bag playback to ensure visualization
        if hasattr(self, 'is_playing') and self.is_playing:
            self.visible = True  # Ensure visibility during playback
        
        # Process point cloud data message
        try:
            points_list = list(pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
            ))
            self.get_logger().debug(f"Received point cloud with {len(points_list)} points")
            
            # Handle empty point cloud gracefully
            if not points_list:
                self.get_logger().debug("Empty point cloud received")
                return
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
                points = list(pc2.read_points(
                    msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
                ))
                self.get_logger().info(f"  - Point cloud contains {len(points)} points")
        
        if not self.visible and not self.collecting_data:
            return  # Skip processing if not needed for visualization or collection

        current_time = time.time()
        if (current_time - self.last_update_time < self.update_interval 
                and not self.collecting_data):
            return  # Rate limiting for visualization updates

        self.last_update_time = current_time

        try:
            with self.data_lock:
                points_list = list(pc2.read_points(
                    msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
                ))
                if not points_list:
                    return

                # Extract and swap x/y for desired visualization orientation
                raw_x = [p[0] for p in points_list]
                raw_y = [p[1] for p in points_list]
                raw_z = [p[2] for p in points_list]
                intensities = [p[3] for p in points_list]

                # Convert to numpy arrays for performance
                x_array = np.array(raw_y, dtype=np.float32)
                y_array = np.array(raw_x, dtype=np.float32)
                z_array = np.array(raw_z, dtype=np.float32)
                intensities_array = np.array(intensities, dtype=np.float32)

                self.current_data['x'] = x_array
                self.current_data['y'] = y_array
                self.current_data['z'] = z_array
                self.current_data['intensities'] = intensities_array

                # Process multi-frame point clouds if enabled
                if self.params.enable_multi_frame:
                    process_multi_frame_data(
                        self, x_array, y_array, z_array, intensities_array
                    )
                
                # Process circle points
                filter_points_in_circle(
                    self, x_array, y_array, intensities_array
                )

                # Decay before adding new data
                apply_live_heatmap_decay(self)

                # Update live heatmap
                update_live_heatmap_vectorized(
                    self, x_array, y_array, intensities_array
                )

                # If collecting data, store new info
                if self.collecting_data and self.collection_start_time is not None:
                    self.process_collected_data(z_array)
                    elapsed_time = time.time() - self.collection_start_time
                    if elapsed_time >= self.params.collection_duration:
                        self.stop_data_collection()

        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

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
        Periodic timer callback for monitoring and maintenance tasks.
        
        This method handles various periodic tasks including:
        - Monitoring bag playback/recording status
        - Checking point cloud message reception
        - Performing diagnostics when needed
        """
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
                            
                            # If we're past the end and still playing, force stop
                            if elapsed_playback > self.bag_duration + 2.0:
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
            self.heatmap_data = np.zeros(calculate_heatmap_size(self.params), dtype=np.float32)

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

            # Save heatmap data
            heatmap_file = os.path.join(
                config_dir,
                f"heatmap_{int(self.params.target_distance)}m_{timestamp}.npz"
            )
            np.savez_compressed(heatmap_file, heatmap=self.heatmap_data)

            self.get_logger().info(f'Saved experiment data to {data_file}')

            # Save time-series data
            if self.experiment_data.time_series_timestamps:
                ts_file = os.path.join(
                    config_dir,
                    f"time_series_{int(self.params.target_distance)}m_{timestamp}.csv"
                )
                df_ts = pd.DataFrame({
                    'timestamp': self.experiment_data.time_series_timestamps,
                    'circle_points': self.experiment_data.circle_point_counts,
                    'circle_avg_intensity': self.experiment_data.circle_avg_intensities
                })
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

            # Calculate distances from origin
            distances = np.sqrt(x_array ** 2 + y_array ** 2)
            
            # Create distance bins
            bins = np.arange(0, self.params.max_range + self.params.circle_interval, self.params.circle_interval)
            counts, _ = np.histogram(distances, bins=bins)

            distance_bands = {}
            for i in range(len(counts)):
                band_key = f"{bins[i]}-{bins[i+1]}m"
                band_mask = (distances >= bins[i]) & (distances < bins[i+1])
                band_intensities = intensities_array[band_mask]
                avg_intensity = float(np.mean(band_intensities)) if band_intensities.size > 0 else 0.0
                distance_bands[band_key] = {'count': float(counts[i]), 'avg_intensity': avg_intensity}

            if self.params.current_config not in self.config_results:
                self.config_results[self.params.current_config] = {}

            # Find the band containing the target distance
            target_band_key = next(
                (band for band in distance_bands
                 if float(band.split('-')[0]) <= self.params.target_distance
                 < float(band.split('-')[1].replace('m', ''))),
                list(distance_bands.keys())[0] if distance_bands else "0-0m"
            )

            circle_counts = len(self.experiment_data.x_points)
            circle_avg_intensity = float(np.mean(intensities_array)) if intensities_array.size > 0 else 0.0

            # Store results in config_results
            results_dict = {
                'total_points': circle_counts,
                'distance_bands': distance_bands,
                'target_band': target_band_key,
                'target_band_points': distance_bands[target_band_key]['count'] if target_band_key in distance_bands else 0,
                'avg_intensity': circle_avg_intensity,
                'circle_points': circle_counts,
                'circle_avg_intensity': circle_avg_intensity
            }
            
            # Add multi-frame metrics if available
            if self.experiment_data.multi_frame_metrics:
                results_dict['multi_frame_metrics'] = self.experiment_data.multi_frame_metrics
            
            self.config_results[self.params.current_config][int(self.params.target_distance)] = results_dict

            self.get_logger().info(
                f'Analyzed data for {self.params.current_config} at {self.params.target_distance}m'
            )
            self.get_logger().info(f'Total points in sampling circle: {circle_counts}')
            self.get_logger().info(f'Average intensity in sampling circle: {circle_avg_intensity:.2f}')
        except Exception as e:
            self.get_logger().error(f"Error analyzing experiment data: {str(e)}")

    def process_collected_data(self, z_array: np.ndarray) -> None:
        """
        Collect and record data from points in all enabled sampling circles for each new frame.
        
        This method adds points within all enabled sampling circles to the experiment data
        and updates the heatmap with these points.

        Args:
            z_array: Z-coordinates of the current point cloud.
        """
        try:
            timestamp = (
                self.get_clock().now().to_msg().sec
                + self.get_clock().now().to_msg().nanosec * 1e-9
            )
            circle_data = ExperimentData()

            # Process data for the primary circle (for backward compatibility)
            indices = self.current_data['circle_indices']
            self.get_logger().debug(f"Processing primary circle: {len(indices)} points")
            
            if len(indices) > 0:
                # Store point data
                circle_data.x_points = self.current_data['circle_x'].tolist()
                circle_data.y_points = self.current_data['circle_y'].tolist()
                circle_data.z_points = z_array[indices].tolist()
                circle_data.intensities = self.current_data['circle_intensities'].tolist()
                circle_data.timestamps = [timestamp] * len(self.current_data['circle_x'])
                circle_data.target_distances = [self.params.target_distance] * len(self.current_data['circle_x'])

                # Calculate statistics
                count = len(self.current_data['circle_x'])
                avg_intensity = (
                    float(np.mean(self.current_data['circle_intensities']))
                    if len(self.current_data['circle_intensities']) > 0
                    else 0
                )
                
                # Store time series data
                circle_data.time_series_timestamps = [timestamp]
                circle_data.circle_point_counts = [count]
                circle_data.circle_avg_intensities = [avg_intensity]

                # Add to experiment data
                self.experiment_data.extend(circle_data)

                # Update heatmap
                update_heatmap_vectorized(
                    self,
                    self.current_data['circle_x'],
                    self.current_data['circle_y'],
                    self.current_data['circle_intensities']
                )
                
            # Process data for secondary circles if enabled
            for i in range(1, len(self.params.circles)):
                if not self.params.circles[i].enabled:
                    self.get_logger().debug(f"Secondary circle {i} is disabled, skipping")
                    continue
                    
                circle_key = f'circle{i+1}'
                secondary_indices = self.current_data[f'{circle_key}_indices']
                self.get_logger().debug(f"Processing secondary circle {i}: {len(secondary_indices)} points")
                
                if len(secondary_indices) > 0:
                    # Create secondary circle data
                    secondary_circle_data = ExperimentData()
                    
                    # Store point data
                    secondary_circle_data.x_points = self.current_data[f'{circle_key}_x'].tolist()
                    secondary_circle_data.y_points = self.current_data[f'{circle_key}_y'].tolist()
                    secondary_circle_data.z_points = z_array[secondary_indices].tolist()
                    secondary_circle_data.intensities = self.current_data[f'{circle_key}_intensities'].tolist()
                    secondary_circle_data.timestamps = [timestamp] * len(self.current_data[f'{circle_key}_x'])
                    secondary_circle_data.target_distances = [self.params.target_distance] * len(self.current_data[f'{circle_key}_x'])
                    
                    # Calculate statistics
                    count = len(self.current_data[f'{circle_key}_x'])
                    avg_intensity = (
                        float(np.mean(self.current_data[f'{circle_key}_intensities']))
                        if len(self.current_data[f'{circle_key}_intensities']) > 0
                        else 0
                    )
                    
                    # Store time series data
                    secondary_circle_data.time_series_timestamps = [timestamp]
                    secondary_circle_data.circle_point_counts = [count]
                    secondary_circle_data.circle_avg_intensities = [avg_intensity]
                    
                    # Store the circle index in the multi_frame_metrics
                    secondary_circle_data.multi_frame_metrics[f'circle_index'] = i
                    
                    # Add to experiment data
                    self.experiment_data.extend(secondary_circle_data)
                    
                    # Update heatmap for this circle too
                    update_heatmap_vectorized(
                        self,
                        self.current_data[f'{circle_key}_x'],
                        self.current_data[f'{circle_key}_y'],
                        self.current_data[f'{circle_key}_intensities']
                    )
                    
                    self.get_logger().debug(f"Added {count} points from circle {i} to experiment data")
                    
        except Exception as e:
            self.get_logger().error(f"Error processing collected data: {str(e)}")
            
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
                    self.current_data[f'circle{i+1}_x'] = np.array([], dtype=np.float32)
                    self.current_data[f'circle{i+1}_y'] = np.array([], dtype=np.float32)
                    self.current_data[f'circle{i+1}_intensities'] = np.array([], dtype=np.float32)
                    self.current_data[f'circle{i+1}_indices'] = np.array([], dtype=np.int32)
                
                # Reset heatmap data
                grid_size = calculate_heatmap_size(self.params)
                self.heatmap_data = np.zeros(grid_size, dtype=np.float32)
                self.live_heatmap_data = np.zeros(grid_size, dtype=np.float32)
                
                # Reset frame buffer if multi-frame processing is enabled
                if self.params.enable_multi_frame:
                    self.frame_buffer = []
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
            
    def play_rosbag(self, bag_path: str, loop: bool = True) -> bool:
        """
        Play a ROS2 bag file.
        
        Wrapper around the play_rosbag function that maintains
        class state for playback operations.
        
        Args:
            bag_path: Path to the ROS2 bag file to play
            loop: Whether to loop the playback when it ends
            
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
        Compute scientific metrics of the live heatmap.
        
        This is a wrapper around the compute_heatmap_metrics function from the
        data_processor module that maintains proper error handling and 
        provides analytics data for the UI.

        Returns:
            Dictionary of relevant metrics including max_intensity, avg_intensity,
            snr_dB, active_cells, total_cells, and coverage_percentage.
        """
        from radar_analyzer.processing.data_processor import compute_heatmap_metrics as compute_metrics_func
        try:
            return compute_metrics_func(self)
        except Exception as e:
            self.get_logger().error(f"Error computing heatmap metrics: {str(e)}")
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
        Update the radius of the sampling circle in both scatter and heatmap.
        
        This is a wrapper around the update_circle_radius function from the
        visualizer module that maintains proper error handling.
        
        Args:
            radius: New radius for the sampling circle in meters.
        """
        from radar_analyzer.visualization.visualizer import update_circle_radius as update_radius_func
        try:
            update_radius_func(self, radius)
        except Exception as e:
            self.get_logger().error(f"Error updating circle radius: {str(e)}")