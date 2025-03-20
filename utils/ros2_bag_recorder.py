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
        
        recording_path = self.current_bag_path
        
        try:
            # First try graceful termination
            logger.info(f"Attempting graceful termination of ROS2 bag recording process {self.recording_process.pid}")
            os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
            
            # Wait with timeout for process to terminate
            try:
                exit_code = self.recording_process.wait(timeout=5)
                logger.info(f"Bag recording stopped with exit code: {exit_code}")
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                logger.warning("Bag recording did not stop gracefully, forcing termination")
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGKILL)
                try:
                    exit_code = self.recording_process.wait(timeout=2)
                    logger.info(f"Bag recording force-killed with exit code: {exit_code}")
                except subprocess.TimeoutExpired:
                    logger.error("Failed to kill bag recording process even with SIGKILL")
                
            # Check for zombie processes using psutil if available
            try:
                import psutil
                parent = psutil.Process(self.recording_process.pid)
                for child in parent.children(recursive=True):
                    logger.warning(f"Killing child process: {child.pid}")
                    child.kill()
                if parent.is_running():
                    logger.warning(f"Killing parent process: {parent.pid}")
                    parent.kill()
            except ImportError:
                logger.warning("psutil not available, skipping additional zombie process checks")
            except psutil.NoSuchProcess:
                logger.info("Main process already terminated (no such process)")
            except Exception as e:
                logger.error(f"Error during psutil cleanup: {e}")
                
            # As a final failsafe, search for any orphaned ros2 bag processes
            self._cleanup_orphaned_processes()
                
        except Exception as e:
            logger.error(f"Error stopping ROS2 bag recording: {e}")
        finally:
            # CRITICAL: Always reset the process reference
            self.recording_process = None
            self.current_bag_path = None
            
        return recording_path
            
    def is_recording(self):
        """
        Check if a recording is in progress.
        
        Returns:
            bool: True if recording is in progress, False otherwise.
        """
        # Double-check if process is actually still running
        if self.recording_process is not None:
            if self.recording_process.poll() is not None:
                # Process has terminated unexpectedly
                logger.warning(f"Recording process terminated unexpectedly with code: {self.recording_process.returncode}")
                self.recording_process = None
                self.current_bag_path = None
                return False
        
        return self.recording_process is not None
        
    def _cleanup_orphaned_processes(self):
        """
        Find and kill any orphaned ros2 bag processes.
        """
        try:
            import psutil
            # Look for any ros2 bag record processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'ros2' in cmdline and 'bag' in cmdline and 'record' in cmdline:
                        logger.warning(f"Found orphaned ROS2 bag process (PID {proc.pid}), killing it")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    logger.debug(f"Could not access process: {e}")
                except Exception as e:
                    logger.error(f"Error checking process: {e}")
        except ImportError:
            logger.warning("psutil not available, cannot check for orphaned processes")
        except Exception as e:
            logger.error(f"Error cleaning up orphaned processes: {e}") 