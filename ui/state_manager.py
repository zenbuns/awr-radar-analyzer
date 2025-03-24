#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application state management for the radar analyzer UI.

This module contains the ApplicationStateManager class which centralizes
state management and UI synchronization.
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)

class ApplicationStateManager:
    """
    Manages application state and UI synchronization.
    
    This class provides centralized state management to ensure UI components
    consistently reflect the actual system state, especially during error
    conditions and asynchronous operations.
    
    Attributes:
        control_panel: Reference to the ControlPanel UI
        states: Dictionary of current application states
    """
    
    def __init__(self, control_panel):
        """
        Initialize the state manager.
        
        Args:
            control_panel: Reference to the ControlPanel UI
        """
        self.control_panel = control_panel
        self.states = {
            'collecting': False,
            'playing_bag': False,
            'recording_bag': False,
            'processing': False,
            'generating_report': False,
            'ui_locked': False  # Global UI lock for critical operations
        }
        
        # Log initial state
        logger.debug("ApplicationStateManager initialized with states: %s", str(self.states))
        
    def transition(self, action, success=True):
        """
        Handle state transitions with UI updates.
        
        This method updates the internal state and synchronizes UI components
        to reflect the new state, providing consistent behavior across the application.
        
        Args:
            action: The action being performed (e.g., 'start_collection')
            success: Whether the action succeeded or failed
            
        Returns:
            bool: Whether the transition was successful
        """
        old_state = self.states.copy()
        
        # State transition logic
        if action == 'start_collection':
            if success:
                self.states['collecting'] = True
            # Update UI based on new state
            self.control_panel.start_button.setEnabled(not self.states['collecting'])
            self.control_panel.stop_button.setEnabled(self.states['collecting'])
            
            # Update ROS2 bag group title and status label
            if hasattr(self.control_panel, 'rosbag_group'):
                self.control_panel.rosbag_group.setTitle("ROS2 Bag Controls - Collecting")
            if hasattr(self.control_panel, 'recording_status_label'):
                self.control_panel.recording_status_label.setText("Collecting Data")
                self.control_panel.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            
        elif action == 'stop_collection':
            self.states['collecting'] = False
            # Update UI based on new state
            self.control_panel.start_button.setEnabled(True)
            self.control_panel.stop_button.setEnabled(False)
            
            # Reset ROS2 bag group title if not collecting
            if not self.states['collecting'] and hasattr(self.control_panel, 'rosbag_group'):
                self.control_panel.rosbag_group.setTitle("ROS2 Bag Controls")
            if hasattr(self.control_panel, 'recording_status_label'):
                self.control_panel.recording_status_label.setText("Collect Settings")
                self.control_panel.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            
        elif action == 'start_playback':
            if success:
                self.states['playing_bag'] = True
            # Update UI based on new state
            self.control_panel.play_button.setEnabled(not self.states['playing_bag'])
            self.control_panel.stop_playback_button.setEnabled(self.states['playing_bag'])
            self.control_panel.timeline_slider.setEnabled(self.states['playing_bag'])
            
            # Update ROS2 bag group title and status label
            if hasattr(self.control_panel, 'rosbag_group'):
                self.control_panel.rosbag_group.setTitle("ROS2 Bag Controls - Playing")
            if hasattr(self.control_panel, 'recording_status_label'):
                self.control_panel.recording_status_label.setText("Record Settings")
                self.control_panel.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            
        elif action == 'stop_playback':
            self.states['playing_bag'] = False
            # Update UI based on new state
            self.control_panel.play_button.setEnabled(True)
            self.control_panel.stop_playback_button.setEnabled(False)
            self.control_panel.timeline_slider.setEnabled(False)
            self.control_panel.timeline_slider.setValue(0)
            
            # Reset ROS2 bag group title if not recording
            if not self.states['recording_bag'] and hasattr(self.control_panel, 'rosbag_group'):
                self.control_panel.rosbag_group.setTitle("ROS2 Bag Controls")
            if hasattr(self.control_panel, 'recording_status_label'):
                self.control_panel.recording_status_label.setText("Record Settings")
                self.control_panel.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            
        elif action == 'start_recording':
            if success:
                self.states['recording_bag'] = True
            # Update UI based on new state
            self.control_panel.record_button.setEnabled(not self.states['recording_bag'])
            self.control_panel.stop_record_button.setEnabled(self.states['recording_bag'])
            
            # Update ROS2 bag group title and status label
            if hasattr(self.control_panel, 'rosbag_group'):
                self.control_panel.rosbag_group.setTitle("ROS2 Bag Controls - Recording")
            if hasattr(self.control_panel, 'recording_status_label'):
                self.control_panel.recording_status_label.setText("Recording in Progress")
                self.control_panel.recording_status_label.setStyleSheet("font-weight: bold; color: #F44336;")
            
        elif action == 'stop_recording':
            self.states['recording_bag'] = False
            # Update UI based on new state
            self.control_panel.record_button.setEnabled(True)
            self.control_panel.stop_record_button.setEnabled(False)
            
            # Reset ROS2 bag group title if not playing
            if not self.states['playing_bag'] and hasattr(self.control_panel, 'rosbag_group'):
                self.control_panel.rosbag_group.setTitle("ROS2 Bag Controls")
            if hasattr(self.control_panel, 'recording_status_label'):
                self.control_panel.recording_status_label.setText("Record Settings")
                self.control_panel.recording_status_label.setStyleSheet("font-weight: bold; color: #757575;")
            
        elif action == 'start_generating_report':
            if success:
                self.states['generating_report'] = True
            # Update UI to show report generation in progress
            if hasattr(self.control_panel, 'generate_report_button'):
                self.control_panel.generate_report_button.setEnabled(not self.states['generating_report'])
                
        elif action == 'stop_generating_report':
            self.states['generating_report'] = False
            # Update UI to show report generation completed
            if hasattr(self.control_panel, 'generate_report_button'):
                self.control_panel.generate_report_button.setEnabled(True)
                
        elif action == 'lock_ui':
            self.states['ui_locked'] = True
            # Disable all interactive controls during critical operations
            self._update_ui_lock()
            
        elif action == 'unlock_ui':
            self.states['ui_locked'] = False
            # Re-enable controls based on other states
            self._update_ui_lock()
            
        # Log state changes for debugging
        changes = {k: v for k, v in self.states.items() if old_state.get(k) != v}
        if changes:
            logger.info(f"State transition: {action} (success={success}) → {changes}")
            
        return success
    
    def get_state(self, state_key):
        """
        Get the current value of a specific state.
        
        Args:
            state_key: Key of the state to retrieve
            
        Returns:
            The current value of the state, or None if the key doesn't exist
        """
        return self.states.get(state_key)
    
    def _update_ui_lock(self):
        """
        Update UI component enabled states based on the global UI lock.
        
        This internal method is called when the UI lock state changes to
        consistently enable/disable all interactive components.
        """
        if self.states['ui_locked']:
            # Disable all interactive controls
            self.control_panel.start_button.setEnabled(False)
            self.control_panel.stop_button.setEnabled(False)
            self.control_panel.play_button.setEnabled(False)
            self.control_panel.stop_playback_button.setEnabled(False)
            self.control_panel.record_button.setEnabled(False)
            self.control_panel.stop_record_button.setEnabled(False)
            self.control_panel.timeline_slider.setEnabled(False)
            if hasattr(self.control_panel, 'generate_report_button'):
                self.control_panel.generate_report_button.setEnabled(False)
        else:
            # Re-enable controls based on current states
            self.control_panel.start_button.setEnabled(not self.states['collecting'])
            self.control_panel.stop_button.setEnabled(self.states['collecting'])
            self.control_panel.play_button.setEnabled(not self.states['playing_bag'])
            self.control_panel.stop_playback_button.setEnabled(self.states['playing_bag'])
            self.control_panel.record_button.setEnabled(not self.states['recording_bag'])
            self.control_panel.stop_record_button.setEnabled(self.states['recording_bag'])
            self.control_panel.timeline_slider.setEnabled(self.states['playing_bag'])
            if hasattr(self.control_panel, 'generate_report_button'):
                self.control_panel.generate_report_button.setEnabled(not self.states['generating_report']) 