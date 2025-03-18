#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS bag utilities for radar point cloud analysis.

This module contains functions for recording, playing, and seeking in
ROS bag files containing radar data.
"""

import os
import time
import signal
import subprocess
import psutil
from typing import List
import numpy as np


def play_rosbag(analyzer, bag_path: str) -> None:
    """Start playback of a ROS2 bag file.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        bag_path: Path to the bag file to play.
        
    Raises:
        RuntimeError: If the bag file doesn't exist or can't be played.
    """
    # Check if the path exists (could be a directory or a .db3 file)
    if not os.path.exists(bag_path):
        error_msg = f"Bag file does not exist: {bag_path}"
        analyzer.get_logger().error(error_msg)
        raise RuntimeError(error_msg)
        
    # Handle case where user selects the .db3 file directly
    if bag_path.endswith('.db3'):
        bag_path = os.path.dirname(bag_path)
        # If bag_path is now empty, use the current directory
        if not bag_path:
            bag_path = '.'
        analyzer.get_logger().info(f"Adjusted bag path to directory: {bag_path}")

    # Verify this is a valid ROS2 bag directory by checking for metadata.yaml
    metadata_path = os.path.join(bag_path, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        error_msg = f"Not a valid ROS2 bag: Missing metadata.yaml in {bag_path}"
        analyzer.get_logger().error(error_msg)
        raise RuntimeError(error_msg)
        
    # Stop any existing bag operations
    if hasattr(analyzer, 'is_recording') and hasattr(analyzer, 'is_playing'):
        if analyzer.is_recording or analyzer.is_playing:
            stop_rosbag(analyzer)
        
    try:
        # First examine the bag to get the topics it contains and duration
        analyzer.get_logger().info(f"Examining topics in bag: {bag_path}")
        try:
            bag_info = subprocess.run(['ros2', 'bag', 'info', bag_path], capture_output=True, text=True)
            analyzer.get_logger().info(f"Bag info:\n{bag_info.stdout}")
            
            # Parse and store bag duration for timeline updates
            duration_line = [line for line in bag_info.stdout.split('\n') if 'Duration:' in line]
            if duration_line:
                duration_str = duration_line[0].split('Duration:')[1].strip().split('s')[0]
                try:
                    analyzer.bag_duration = float(duration_str)
                    analyzer.get_logger().info(f"Bag duration: {analyzer.bag_duration:.2f} seconds")
                except ValueError:
                    analyzer.get_logger().error(f"Failed to parse bag duration: {duration_str}")
                    analyzer.bag_duration = 0.0
            else:
                analyzer.bag_duration = 0.0
        except Exception as e:
            analyzer.get_logger().warn(f"Could not get bag info: {str(e)}")
            analyzer.bag_duration = 0.0
        
        # Prepare ros2 bag play command with enhanced options for better compatibility
        cmd = [
            'ros2', 'bag', 'play',
            '--loop',  # Loop playback for testing
            '--read-ahead-queue-size', '1000',  # Increase buffer size
            bag_path
        ]
        
        analyzer.get_logger().info(f"Starting ROS2 bag playback with command: {' '.join(cmd)}")
        
        analyzer.rosbag_proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # To make it a process group leader
        )
        
        analyzer.is_playing = True
        analyzer.current_bag_path = bag_path
        analyzer.bag_start_time = time.time()
        
        analyzer.get_logger().info(f"Started playing ROS2 bag: {bag_path}")
    except Exception as e:
        error_msg = f"Failed to play ROS2 bag: {str(e)}"
        analyzer.get_logger().error(error_msg)
        raise RuntimeError(error_msg)


def record_rosbag(analyzer, output_path: str, topics: List[str]) -> None:
    """Start recording a ROS2 bag file.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        output_path: Path where the bag should be saved.
        topics: List of topics to record.
        
    Raises:
        RuntimeError: If recording can't be started.
    """
    # Stop any existing bag operations
    if hasattr(analyzer, 'is_recording') and hasattr(analyzer, 'is_playing'):
        if analyzer.is_recording or analyzer.is_playing:
            stop_rosbag(analyzer)
        
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        # Format topics for ros2 bag record command
        topics_args = topics if topics else ['-a']  # Record all topics if none specified
        
        # Create the ros2 bag record command
        cmd = ['ros2', 'bag', 'record', '-o', output_path] + topics_args
        
        analyzer.get_logger().info(f"Starting ROS2 bag recording with command: {' '.join(cmd)}")
        
        # Start the recording process
        analyzer.rosbag_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # To make it a process group leader
        )
        
        analyzer.is_recording = True
        analyzer.current_bag_path = output_path
        analyzer.bag_start_time = time.time()
        
        analyzer.get_logger().info(f"Started recording ROS2 bag to: {output_path}")
    except Exception as e:
        error_msg = f"Failed to start ROS2 bag recording: {str(e)}"
        analyzer.get_logger().error(error_msg)
        raise RuntimeError(error_msg)


def stop_rosbag(analyzer) -> None:
    """
    Stop ROS2 bag playback or recording and perform thorough cleanup.
    
    This method ensures all ROS2 bag processes are completely terminated,
    message queues are cleared, and the analyzer is reset to a clean state.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
    """
    # Reset state flags first to prevent processing of new messages during cleanup
    analyzer.is_recording = False
    analyzer.is_playing = False
    analyzer.visible = False
    
    # Track cleanup success
    cleanup_success = True
    
    # 1. Kill the primary rosbag process if it exists
    if hasattr(analyzer, 'rosbag_proc') and analyzer.rosbag_proc is not None:
        try:
            # Get the process group ID before killing
            pgid = os.getpgid(analyzer.rosbag_proc.pid)
            
            # Terminate the process group to ensure all child processes are killed
            analyzer.get_logger().info(f"Sending SIGINT to process group {pgid}")
            os.killpg(pgid, signal.SIGINT)
            
            # Wait a short time for graceful termination
            try:
                analyzer.rosbag_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                analyzer.get_logger().warn(f"Sending SIGKILL to process group {pgid}")
                os.killpg(pgid, signal.SIGKILL)
            
            # Clean up any zombie processes
            try:
                # Use psutil to ensure all child processes are terminated
                parent = psutil.Process(analyzer.rosbag_proc.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except psutil.NoSuchProcess:
                pass  # Process already terminated
            
            analyzer.rosbag_proc = None
        except Exception as e:
            analyzer.get_logger().error(f"Error stopping primary ROS2 bag process: {str(e)}")
            cleanup_success = False
    
    # 2. Find and kill any other ros2 bag processes that might be running
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'ros2' in cmdline and 'bag' in cmdline:
                    analyzer.get_logger().warn(f"Killing residual ROS2 bag process: {proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        analyzer.get_logger().error(f"Error killing residual ROS2 bag processes: {str(e)}")
        cleanup_success = False
    
    # 3. Reset subscriptions to clear any queued messages
    try:
        # Destroy existing subscriptions
        if hasattr(analyzer, 'pcl_subscription'):
            analyzer.destroy_subscription(analyzer.pcl_subscription)
        if hasattr(analyzer, 'track_array_subscription'):
            analyzer.destroy_subscription(analyzer.track_array_subscription)
        if hasattr(analyzer, 'occupancy_subscription'):
            analyzer.destroy_subscription(analyzer.occupancy_subscription)
        
        # Recreate subscriptions with the same reliable QoS profile
        analyzer.pcl_subscription = analyzer.create_subscription(
            analyzer.__class__._pcl_msg_type, 
            '/ti_mmwave/radar_scan_pcl', 
            analyzer.pcl_callback, 
            analyzer.reliable_qos
        )
        analyzer.track_array_subscription = analyzer.create_subscription(
            analyzer.__class__._track_array_msg_type, 
            '/ti_mmwave/radar_track_marker_array', 
            analyzer.track_array_callback, 
            analyzer.reliable_qos
        )
        analyzer.occupancy_subscription = analyzer.create_subscription(
            analyzer.__class__._occupancy_msg_type, 
            '/ti_mmwave/radar_occupancy', 
            analyzer.occupancy_callback, 
            analyzer.reliable_qos
        )
        analyzer.get_logger().info("Reset all subscriptions to clear message queues")
    except Exception as e:
        analyzer.get_logger().error(f"Error resetting subscriptions: {str(e)}")
        cleanup_success = False
    
    # 4. Perform hard reset of PCL data
    try:
        # Clear hard reset first
        analyzer.hard_reset_pcl()
        
        # Force a UI update to show the cleared data
        try:
            # Make sure visualization knows about the reset
            if hasattr(analyzer, 'signals') and hasattr(analyzer.signals, 'data_reset_signal'):
                analyzer.signals.data_reset_signal.emit()
                analyzer.get_logger().info("Emitted data reset signal to UI")
        except Exception as e:
            analyzer.get_logger().error(f"Error emitting data reset signal: {str(e)}")
    except Exception as e:
        analyzer.get_logger().error(f"Error during PCL hard reset: {str(e)}")
        cleanup_success = False
    
    # 5. Stop data collection if it was active
    if analyzer.collecting_data:
        try:
            analyzer.stop_data_collection()
        except Exception as e:
            analyzer.get_logger().error(f"Error stopping data collection: {str(e)}")
            cleanup_success = False
    
    # 6. Reset all state variables and clear any remaining data
    analyzer.current_bag_path = None
    analyzer.bag_start_time = None
    analyzer.pcl_msg_count = 0
    analyzer.last_pcl_msg_time = None
    analyzer.bag_duration = 0.0  # Reset bag duration
    analyzer.last_position = -1.0  # Reset last position
    analyzer.last_position_update_time = 0  # Reset last position update time
    
    # 7. Clear any remaining ROS2 topics
    try:
        # Use ros2 topic command to clear any remaining messages
        subprocess.run(['ros2', 'topic', 'echo', '/ti_mmwave/radar_scan_pcl', '--once'], 
                     capture_output=True, timeout=1.0)
        subprocess.run(['ros2', 'topic', 'echo', '/ti_mmwave/radar_track_marker_array', '--once'], 
                     capture_output=True, timeout=1.0)
        subprocess.run(['ros2', 'topic', 'echo', '/ti_mmwave/radar_occupancy', '--once'], 
                     capture_output=True, timeout=1.0)
    except Exception as e:
        analyzer.get_logger().debug(f"Error clearing ROS2 topics: {str(e)}")
    
    # 8. Update the UI to reflect the reset
    try:
        # Force a visualization update to clear the displays
        if hasattr(analyzer, 'viz_components') and analyzer.viz_components.get('fig') is not None:
            for key in ['scatter', 'circle_scatter']:
                if analyzer.viz_components.get(key) is not None:
                    analyzer.viz_components[key].set_offsets(np.empty((0, 2)))
                    if hasattr(analyzer.viz_components[key], 'set_array'):
                        analyzer.viz_components[key].set_array(np.array([]))
            
            # Clear text displays
            for key in ['stats_text', 'circle_stats_text']:
                if analyzer.viz_components.get(key) is not None:
                    analyzer.viz_components[key].set_text("")
            
            # Reset heatmap if it exists
            if hasattr(analyzer, 'heatmap_viz') and analyzer.heatmap_viz.get('heatmap') is not None:
                grid_size = hasattr(analyzer, 'params') and hasattr(analyzer.params, 'heatmap_resolution')
                if grid_size:
                    empty_grid = np.zeros_like(analyzer.heatmap_viz['heatmap'].get_array())
                    analyzer.heatmap_viz['heatmap'].set_data(empty_grid)
                
                # Clear contours if they exist
                if analyzer.heatmap_viz.get('contour') is not None:
                    for coll in analyzer.heatmap_viz['contour'].collections:
                        try:
                            coll.remove()
                        except Exception:
                            pass
                    analyzer.heatmap_viz['contour'] = None
                
                # Reset SNR text
                if analyzer.heatmap_viz.get('snr_text') is not None:
                    analyzer.heatmap_viz['snr_text'].set_text("SNR: N/A")
            
            # Draw the changes
            try:
                if hasattr(analyzer.viz_components['fig'], 'canvas'):
                    analyzer.viz_components['fig'].canvas.draw_idle()
            except Exception as e:
                analyzer.get_logger().error(f"Error updating visualization: {str(e)}")
    except Exception as e:
        analyzer.get_logger().error(f"Error updating UI after reset: {str(e)}")
        cleanup_success = False
    
    if cleanup_success:
        analyzer.get_logger().info("Successfully stopped ROS2 bag and completely reset analyzer state")
    else:
        analyzer.get_logger().warn("Stopped ROS2 bag with some cleanup errors - analyzer reset may be incomplete")


def seek_rosbag(analyzer, position: float) -> None:
    """Seek to a specific position in the currently playing ROS2 bag.
    
    This method stops and restarts playback with the --start-offset parameter
    to seek to a specific position in the bag file.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        position: Normalized position in the bag (0.0-1.0)
    """
    if (not hasattr(analyzer, 'is_playing') or not analyzer.is_playing or 
            not hasattr(analyzer, 'current_bag_path') or analyzer.current_bag_path is None):
        analyzer.get_logger().warn("Cannot seek: No ROS2 bag is currently playing")
        return
        
    # Validate position bounds
    position = max(0.0, min(1.0, position))
    
    try:
        # If we already know the bag duration, we don't need to query it again
        duration = getattr(analyzer, 'bag_duration', None)
        if duration is None or duration <= 0:
            # Need to get the bag duration first
            try:
                bag_info = subprocess.run(['ros2', 'bag', 'info', analyzer.current_bag_path], 
                                         capture_output=True, text=True, timeout=2.0)
                # Parse duration from the output (format: "Duration: 10.123s")
                duration_line = [line for line in bag_info.stdout.split('\n') 
                               if 'Duration:' in line]
                if duration_line:
                    duration_str = duration_line[0].split('Duration:')[1].strip().split('s')[0]
                    try:
                        duration = float(duration_str)
                        # Store duration for future use
                        analyzer.bag_duration = duration
                    except ValueError:
                        analyzer.get_logger().error(f"Failed to parse bag duration: {duration_str}")
                        duration = None
            except Exception as e:
                analyzer.get_logger().error(f"Error determining bag duration: {str(e)}")
                duration = None
        
        # Calculate offset in seconds if possible
        if duration is not None and duration > 0:
            offset = position * duration
            analyzer.get_logger().info(f"Seeking to {offset:.2f}s in a {duration:.2f}s bag")
            
            # Stop current playback gracefully but keep the 'is_playing' state
            if hasattr(analyzer, 'rosbag_proc') and analyzer.rosbag_proc is not None:
                try:
                    os.killpg(os.getpgid(analyzer.rosbag_proc.pid), signal.SIGTERM)
                    # Wait very briefly to allow termination
                    try:
                        analyzer.rosbag_proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        # If it doesn't terminate quickly, force it
                        os.killpg(os.getpgid(analyzer.rosbag_proc.pid), signal.SIGKILL)
                except Exception as e:
                    analyzer.get_logger().warn(f"Error stopping previous playback: {str(e)}")
            
            # Start new playback with the offset
            cmd = [
                'ros2', 'bag', 'play',
                '--loop',
                '--read-ahead-queue-size', '1000',
                '--start-offset', f"{offset:.2f}",
                analyzer.current_bag_path
            ]
            
            analyzer.get_logger().info(f"Starting ROS2 bag playback with seek: {' '.join(cmd)}")
            
            analyzer.rosbag_proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )
            
            # Adjust the start time to account for the seek position
            analyzer.bag_start_time = time.time() - offset
            
            # Immediately update the UI with the new position
            try:
                # Use the signals object to emit the position update
                analyzer.signals.update_playback_position_signal.emit(position)
            except Exception as e:
                analyzer.get_logger().debug(f"Error emitting position update: {str(e)}")
            
            return
        
        # Fallback: Just restart from beginning
        analyzer.get_logger().info(f"Restarting bag from beginning: {analyzer.current_bag_path}")
        play_rosbag(analyzer, analyzer.current_bag_path)
        
    except Exception as e:
        analyzer.get_logger().error(f"Error seeking in ROS2 bag: {str(e)}")
