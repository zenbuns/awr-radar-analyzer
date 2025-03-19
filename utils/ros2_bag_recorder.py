#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 bag recording functionality for the radar analyzer.

This module provides the ability to record ROS2 bags during
radar data collection sessions.
"""

import os
import subprocess
import signal
import datetime
import logging

logger = logging.getLogger(__name__)

class ROS2BagRecorder:
    """
    Records ROS2 bag files during data collection.
    
    This class manages the recording of ROS2 bag files, starting
    when data collection begins and stopping when it ends.
    """
    
    def __init__(self, base_dir='data'):
        """
        Initialize the ROS2 bag recorder.
        
        Args:
            base_dir: Base directory for storing bag files. 
                      Should be the same as where experimental data is saved.
        """
        self.base_dir = base_dir
        self.recording_process = None
        self.current_bag_path = None
    
    def start_recording(self, config_name, target_distance):
        """
        Start recording a ROS2 bag.
        
        Args:
            config_name: Name of the current configuration.
            target_distance: Target distance being measured.
            
        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        if self.recording_process:
            logger.warning("Attempted to start recording when already recording")
            return False
        
        # Create timestamp for unique bag naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory if it doesn't exist
        config_dir = os.path.join(self.base_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        # Define bag file path
        bag_name = f"{config_name}_{target_distance}m_{timestamp}"
        self.current_bag_path = os.path.join(config_dir, bag_name)
        
        try:
            # Start ros2 bag record command for all topics
            self.recording_process = subprocess.Popen(
                ["ros2", "bag", "record", "-a", "-o", self.current_bag_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Used to send signal to the process group
            )
            
            logger.info(f"Started ROS2 bag recording to {self.current_bag_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ROS2 bag recording: {e}")
            self.current_bag_path = None
            return False
    
    def stop_recording(self):
        """
        Stop the current ROS2 bag recording.
        
        Returns:
            str: Path to the recorded bag file or None if recording failed.
        """
        if not self.recording_process:
            logger.warning("Attempted to stop recording when not recording")
            return None
        
        try:
            # Terminate the recording process and all its children
            os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
            
            # Wait for process to terminate
            self.recording_process.wait(timeout=5)
            
            logger.info(f"Stopped ROS2 bag recording to {self.current_bag_path}")
            bag_path = self.current_bag_path
            
            # Clean up
            self.recording_process = None
            self.current_bag_path = None
            
            return bag_path
            
        except Exception as e:
            logger.error(f"Error stopping ROS2 bag recording: {e}")
            
            # Force kill if needed
            try:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGKILL)
            except:
                pass
                
            self.recording_process = None
            self.current_bag_path = None
            return None
            
    def is_recording(self):
        """
        Check if a recording is in progress.
        
        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        return self.recording_process is not None 