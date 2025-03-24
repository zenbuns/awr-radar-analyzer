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
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool


def play_rosbag(analyzer, bag_path: str, loop: bool = False) -> None:
    """Start playback of a ROS2 bag file.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        bag_path: Path to the bag file to play.
        loop: Whether to loop the playback or play once. Defaults to False.
        
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
    
    # Clear experiment data to prevent accumulation from previous runs
    with analyzer.data_lock:
        if hasattr(analyzer, 'experiment_data'):
            analyzer.experiment_data.clear()
            analyzer.get_logger().info("Cleared experiment data for new bag playback")
            
    # Set visibility to true for data collection purposes
    analyzer.visible = True
            
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
            '--read-ahead-queue-size', '1000',  # Increase buffer size
        ]
        
        # Only add loop option if requested
        if loop:
            cmd.append('--loop')  # Loop playback for testing
            analyzer.get_logger().info("Bag will loop when playback ends")
        else:
            analyzer.get_logger().info("Bag will play once and then stop")
        
        # Add bag path last
        cmd.append(bag_path)
        
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
        
        # Store whether this is a looping playback
        analyzer.bag_looping = loop
        
        analyzer.get_logger().info(f"Started playing ROS2 bag: {bag_path}")
    except Exception as e:
        error_msg = f"Failed to play ROS2 bag: {str(e)}"
        analyzer.get_logger().error(error_msg)
        raise RuntimeError(error_msg)


def record_rosbag(analyzer, output_path: str, topics: List[str], duration_minutes: int = 0) -> None:
    """Start recording a ROS2 bag file.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
        output_path: Path where the bag should be saved.
        topics: List of topics to record.
        duration_minutes: Optional recording duration in minutes (0 = no limit).
        
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
        
        # If duration is specified, convert to seconds and add to command
        if duration_minutes > 0:
            duration_seconds = duration_minutes * 60
            cmd.extend(['--duration', str(duration_seconds)])
            analyzer.get_logger().info(f"Setting recording duration to {duration_minutes} minutes ({duration_seconds} seconds)")
        
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
        
        # Store the expected recording end time if duration is set
        if duration_minutes > 0:
            analyzer.recording_end_time = time.time() + (duration_minutes * 60)
        else:
            analyzer.recording_end_time = None
        
        analyzer.get_logger().info(f"Started recording ROS2 bag to: {output_path}")
    except Exception as e:
        error_msg = f"Failed to start ROS2 bag recording: {str(e)}"
        analyzer.get_logger().error(error_msg)
        raise RuntimeError(error_msg)


def stop_rosbag(analyzer) -> None:
    """Stop the currently playing or recording ROS2 bag.
    
    This method terminates any active ROS2 bag playback or recording,
    and cleans up related resources.
    
    Args:
        analyzer: RadarPointCloudAnalyzer instance.
    """
    # Keep track of previous state for signaling
    was_recording = analyzer.is_recording
    was_playing = analyzer.is_playing
    
    # Reset state flags first to prevent processing of new messages during cleanup
    analyzer.is_recording = False
    analyzer.is_playing = False
    analyzer.visible = False
    
    # Track cleanup success
    cleanup_success = True
    
    # 1. Kill the primary rosbag process if it exists
    if hasattr(analyzer, 'rosbag_proc') and analyzer.rosbag_proc is not None:
        try:
            # Get process info before killing
            pid = analyzer.rosbag_proc.pid
            
            # Get the process group ID before killing
            pgid = os.getpgid(pid)
            
            analyzer.get_logger().info(f"Stopping ROS2 bag process {pid} (group {pgid})")
            
            # Terminate the process group to ensure all child processes are killed
            analyzer.get_logger().info(f"Sending SIGINT to process group {pgid}")
            os.killpg(pgid, signal.SIGINT)
            
            # Wait a short time for graceful termination
            try:
                exit_code = analyzer.rosbag_proc.wait(timeout=3)
                analyzer.get_logger().info(f"ROS2 bag process {pid} terminated with exit code: {exit_code}")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                analyzer.get_logger().warn(f"Sending SIGKILL to process group {pgid}")
                os.killpg(pgid, signal.SIGKILL)
                
                try:
                    exit_code = analyzer.rosbag_proc.wait(timeout=2)
                    analyzer.get_logger().info(f"ROS2 bag process {pid} killed with exit code: {exit_code}")
                except subprocess.TimeoutExpired:
                    analyzer.get_logger().error(f"Failed to kill ROS2 bag process {pid} even with SIGKILL")
            
            # Clean up any zombie processes
            try:
                # Use psutil to ensure all child processes are terminated
                import psutil
                try:
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        analyzer.get_logger().warn(f"Killing child process: {child.pid}")
                        child.kill()
                    if parent.is_running():
                        analyzer.get_logger().warn(f"Killing parent process: {parent.pid}")
                        parent.kill()
                except psutil.NoSuchProcess:
                    analyzer.get_logger().info(f"Process {pid} already terminated")
            except ImportError:
                analyzer.get_logger().warn("psutil not available, skipping zombie process checks")
            except Exception as e:
                analyzer.get_logger().error(f"Error during process cleanup: {str(e)}")
            
        except Exception as e:
            analyzer.get_logger().error(f"Error stopping primary ROS2 bag process: {str(e)}")
            cleanup_success = False
        finally:
            # CRITICAL: Always clear the process reference
            analyzer.rosbag_proc = None
    
    # 2. Find and kill any other ros2 bag processes that might be running
    try:
        import psutil
        cleaned_orphans = False
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'ros2' in cmdline and 'bag' in cmdline:
                    analyzer.get_logger().warn(f"Killing orphaned ROS2 bag process: {proc.info['pid']}")
                    proc.kill()
                    cleaned_orphans = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except Exception as e:
                analyzer.get_logger().error(f"Error killing process: {str(e)}")
        
        if cleaned_orphans:
            analyzer.get_logger().info("Cleaned up orphaned ROS2 bag processes")
    except ImportError:
        analyzer.get_logger().warn("psutil not available, cannot check for orphaned processes")
    except Exception as e:
        analyzer.get_logger().error(f"Error killing residual ROS2 bag processes: {str(e)}")
        
    # Clear circle ROI data to prevent stale data in visualization
    try:
        with analyzer.data_lock:
            # Clear current frame data
            analyzer.current_data['circle_x'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_y'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_intensities'] = np.array([], dtype=np.float32)
            analyzer.current_data['circle_indices'] = np.array([], dtype=np.int32)
            
            # Clear additional circles if they exist
            for i in range(1, len(analyzer.params.circles) if hasattr(analyzer.params, 'circles') else 0):
                circle_key = f'circle{i+1}'
                analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
    except Exception as e:
        analyzer.get_logger().error(f"Error clearing circle ROI data: {str(e)}")
        
    # Final cleanup and notification
    analyzer.get_logger().info("ROS2 bag stopped and data cleared")
    
    # Emit the appropriate signal based on what was previously active
    try:
        if (was_playing or was_recording) and hasattr(analyzer, 'signals'):
            # Notify the UI that playback/recording has ended
            analyzer.signals.bag_playback_ended.emit()
    except Exception as e:
        analyzer.get_logger().error(f"Error emitting bag_playback_ended signal: {str(e)}")


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
        # Clear circle ROI data before seeking to prevent stale data
        try:
            with analyzer.data_lock:
                # Clear current frame data
                analyzer.current_data['circle_x'] = np.array([], dtype=np.float32)
                analyzer.current_data['circle_y'] = np.array([], dtype=np.float32)
                analyzer.current_data['circle_intensities'] = np.array([], dtype=np.float32)
                analyzer.current_data['circle_indices'] = np.array([], dtype=np.int32)
                
                # Clear additional circles if they exist
                for i in range(1, len(analyzer.params.circles) if hasattr(analyzer.params, 'circles') else 0):
                    circle_key = f'circle{i+1}'
                    analyzer.current_data[f'{circle_key}_x'] = np.array([], dtype=np.float32)
                    analyzer.current_data[f'{circle_key}_y'] = np.array([], dtype=np.float32)
                    analyzer.current_data[f'{circle_key}_intensities'] = np.array([], dtype=np.float32)
                    analyzer.current_data[f'{circle_key}_indices'] = np.array([], dtype=np.int32)
                    
            analyzer.get_logger().info("Cleared circle ROI data before seeking")
        except Exception as e:
            analyzer.get_logger().error(f"Error clearing circle ROI data before seeking: {str(e)}")
            
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
            
            # Check whether to loop the playback
            loop = getattr(analyzer, 'bag_looping', False)  # Default to False for backward compatibility
            
            # Start new playback with the offset
            cmd = [
                'ros2', 'bag', 'play',
                '--read-ahead-queue-size', '1000',
                '--start-offset', f"{offset:.2f}",
            ]
            
            # Add loop option if needed
            if loop:
                cmd.append('--loop')
                
            # Add bag path last
            cmd.append(analyzer.current_bag_path)
            
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
        loop = getattr(analyzer, 'bag_looping', False)  # Default to False for backward compatibility
        play_rosbag(analyzer, analyzer.current_bag_path, loop)
        
    except Exception as e:
        analyzer.get_logger().error(f"Error seeking in ROS2 bag: {str(e)}")
