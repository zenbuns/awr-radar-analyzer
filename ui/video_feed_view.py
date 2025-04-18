#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt-based video feed view.

This module provides a widget that displays a video feed from a ROS2 topic.
"""

import sys
import os

import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

# Import ROS2 dependencies with error handling
CV_BRIDGE_AVAILABLE = False
ROS2_AVAILABLE = False
CvBridge = None  # Define CvBridge as None initially

try:
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
    
    # Try to import cv_bridge safely, catching both ImportError and the specific AttributeError
    try:
        import importlib.util
        cv_bridge_spec = importlib.util.find_spec("cv_bridge")
        if cv_bridge_spec:
            try:
                # This is the line that can raise AttributeError with NumPy 2.0
                from cv_bridge import CvBridge as CvBridge_import
                CvBridge = CvBridge_import # Assign to the global scope only if import succeeds
                CV_BRIDGE_AVAILABLE = True
                print("cv_bridge imported successfully.")
            except AttributeError as e:
                if "_ARRAY_API not found" in str(e):
                    print("NumPy 2.0 compatibility issue detected with cv_bridge.")
                    print("Using manual image conversion as fallback.")
                else:
                    # Re-raise unexpected AttributeErrors
                    raise e 
                CV_BRIDGE_AVAILABLE = False
            except ImportError as e:
                 print(f"CV_Bridge import error: {e}")
                 CV_BRIDGE_AVAILABLE = False
        else:
             print("cv_bridge package not found.")
             CV_BRIDGE_AVAILABLE = False
             
    except Exception as e:
        # Catch any other potential issues during the import process
        print(f"Error checking for cv_bridge: {e}")
        CV_BRIDGE_AVAILABLE = False
except ImportError as e:
    print(f"ROS2 core dependencies import error: {e}")


class VideoFeedView(QWidget):
    """
    A widget that displays a video feed from a ROS2 topic.
    
    This widget uses ROS2 to subscribe to an Image topic and displays the video
    feed.
    
    Attributes:
        analyzer: RadarPointCloudAnalyzer node for ROS2 subscriptions.
        bridge: CvBridge for converting ROS2 images to OpenCV format.
        latest_frame: Latest frame received from the video feed.
        frame_received: Flag indicating if a frame has been received.
        subscription: ROS2 subscription to the video feed topic.
        video_label: QLabel for displaying the video feed.
        update_timer: QTimer for updating the display.
    """
    
    def __init__(self, parent=None, analyzer=None):
        """
        Initialize the video feed view.
        
        Args:
            parent: Parent widget (optional).
            analyzer: RadarPointCloudAnalyzer node for ROS2 subscriptions (optional).
        """
        super().__init__(parent)
        
        # Declare globals that will be modified
        global CV_BRIDGE_AVAILABLE, ROS2_AVAILABLE
        
        # Store reference to analyzer node
        self.analyzer = analyzer
        
        # Set up ROS subscriber if available
        self.bridge = None
        self.latest_frame = None
        self.frame_received = False
        self.subscription = None
        
        if ROS2_AVAILABLE and CV_BRIDGE_AVAILABLE and CvBridge:
            try:
                self.bridge = CvBridge()
                # Create subscription in init_ros if analyzer exists
                if self.analyzer:
                    self.init_ros()
            except Exception as e:
                print(f"Error initializing CvBridge: {e}")
                CV_BRIDGE_AVAILABLE = False
        
        # Set up UI
        self.setup_ui()
        
        # Initialize with a test pattern if no video feed is available
        if not ROS2_AVAILABLE or not CV_BRIDGE_AVAILABLE:
            self.create_test_pattern()
            # Ensure the initial test pattern is displayed
            self.update_display()
        
    def create_test_pattern(self):
        """Create a test pattern image when no real video feed is available."""
        # Create a test pattern image (color bars)
        height, width = 480, 640
        self.latest_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create color bars
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 255) # White
        ]
        
        bar_width = width // len(colors)
        for i, color in enumerate(colors):
            x_start = i * bar_width
            x_end = (i + 1) * bar_width if i < len(colors) - 1 else width
            self.latest_frame[:, x_start:x_end] = color
            
        # Add text
        cv2.putText(
            self.latest_frame, 
            "No ROS2 Video Feed Available", 
            (width // 6, height // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 0), 
            2
        )
        
        if not ROS2_AVAILABLE:
            cv2.putText(
                self.latest_frame, 
                "ROS2 is not installed", 
                (width // 6, height // 2 + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 0), 
                2
            )
        elif not CV_BRIDGE_AVAILABLE:
            cv2.putText(
                self.latest_frame, 
                "CV_Bridge error - NumPy 2.0 compatibility issue", 
                (width // 6, height // 2 + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 0), 
                2
            )
        
        self.frame_received = True
        
    def setup_ui(self):
        """Set up the UI components."""
        # Main layout
        layout = QVBoxLayout(self) # Changed to QVBoxLayout as it only contains the video now
        layout.setContentsMargins(0, 0, 0, 0) # Removed margins for a cleaner look
        
        # Video feed label
        self.video_label = QLabel("Waiting for video feed...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(640, 480) # Keep minimum size
        self.video_label.setStyleSheet("background-color: #1E1E1E; color: white;")
        layout.addWidget(self.video_label)
        
        # Update timer - Consider moving update logic directly into callback
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(33)  # ~30 FPS, might be inefficient if no new frames
        
    def init_ros(self):
        """Initialize ROS2 subscription."""
        if self.analyzer and ROS2_AVAILABLE:
            try:
                # Create subscription to video topic
                self.subscription = self.analyzer.create_subscription(
                    Image,
                    '/out',  # Use the /out topic as requested
                    self.image_callback,
                    10  # QoS profile depth
                )
                print(f"Subscribed to ROS2 topic: /out")
            except Exception as e:
                print(f"Error creating ROS2 subscription: {e}")
    
    def imgmsg_to_cv2_manual(self, img_msg):
        """
        Manual implementation of cv_bridge's imgmsg_to_cv2 function for NumPy 2.0 compatibility
        """
        if not hasattr(img_msg, 'encoding') or not hasattr(img_msg, 'data') or not hasattr(img_msg, 'height') or not hasattr(img_msg, 'width') or not hasattr(img_msg, 'step'):
            print("Invalid image message format")
            return None
            
        # Extract message data
        encoding = img_msg.encoding
        data = np.frombuffer(img_msg.data, dtype=np.uint8)
        height = img_msg.height
        width = img_msg.width
        step = img_msg.step
        is_bigendian = img_msg.is_bigendian if hasattr(img_msg, 'is_bigendian') else False
        
        # Handle different encodings
        if encoding in ["bgr8", "rgb8"]:
            channels = 3
            # Reshape data to image dimensions
            try:
                # Check if step is correct for the expected image size
                if step >= width * channels:
                    # Use step to handle any padding
                    image = np.zeros((height, width, channels), dtype=np.uint8)
                    for i in range(height):
                        row_data = data[i*step:i*step + width*channels]
                        if len(row_data) >= width*channels:
                            row = row_data[:width*channels].reshape(width, channels)
                            image[i, :, :] = row
                else:
                    # Fallback to simple reshape if step doesn't match expected size
                    image = data.reshape((height, width, channels))
                
                # Convert RGB to BGR if needed
                if encoding == "rgb8":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                # Swap byte order if needed
                if is_bigendian and (sys.byteorder == 'little'):
                    image = image.byteswap().newbyteorder()
                    
                return image
            except Exception as e:
                print(f"Error reshaping image data: {e}")
                return None
                
        elif encoding == "mono8":
            # Reshape data to image dimensions
            try:
                # Check if step is correct for the expected image size
                if step >= width:
                    # Use step to handle any padding
                    image = np.zeros((height, width), dtype=np.uint8)
                    for i in range(height):
                        row_data = data[i*step:i*step + width]
                        if len(row_data) >= width:
                            image[i, :] = row_data[:width]
                else:
                    # Fallback to simple reshape
                    image = data.reshape((height, width))
                
                # Swap byte order if needed
                if is_bigendian and (sys.byteorder == 'little'):
                    image = image.byteswap().newbyteorder()
                    
                return image
            except Exception as e:
                print(f"Error reshaping image data: {e}")
                return None
                
        elif encoding == "16UC1" or encoding == "mono16":
            # Handle 16-bit mono images
            try:
                # Convert 8-bit array to 16-bit
                data_16 = np.frombuffer(img_msg.data, dtype=np.uint16)
                
                # Reshape according to step
                if step >= width * 2:  # 2 bytes per pixel for 16-bit
                    # Use step to handle any padding
                    image = np.zeros((height, width), dtype=np.uint16)
                    for i in range(height):
                        row_start = (i * step) // 2  # Divide by 2 since data_16 elements are 2 bytes
                        row_end = row_start + width
                        if row_end <= len(data_16):
                            image[i, :] = data_16[row_start:row_end]
                else:
                    # Fallback to simple reshape
                    image = data_16.reshape((height, width))
                
                # Swap byte order if needed
                if is_bigendian and (sys.byteorder == 'little'):
                    image = image.byteswap().newbyteorder()
                    
                return image
            except Exception as e:
                print(f"Error processing 16-bit image: {e}")
                return None
        else:
            print(f"Unsupported encoding: {encoding}")
            return None
        
    def image_callback(self, msg):
        """Callback for when a new image is received."""
        if not ROS2_AVAILABLE:
            return
            
        cv_image = None
        try:
            # Try using cv_bridge ONLY if it's available and initialized
            if CV_BRIDGE_AVAILABLE and self.bridge:
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    self.latest_frame = cv_image
                    self.frame_received = True
                    return # Successfully converted with cv_bridge
                except Exception as bridge_error:
                    print(f'Error using cv_bridge: {bridge_error}')
                    # If cv_bridge fails, proceed to manual conversion below
                    pass 
            
            # Use manual conversion if cv_bridge is not available or failed
            cv_image = self.imgmsg_to_cv2_manual(msg)
            if cv_image is not None:
                self.latest_frame = cv_image
                self.frame_received = True
                # Optional: Trigger update directly here for efficiency
                # QTimer.singleShot(0, self.update_display) 
                
        except Exception as e:
            print(f'Error converting image: {e}')
            
    def update_display(self):
        """Update the video feed display."""
        # Update video display
        if self.frame_received and self.latest_frame is not None:
            frame = self.latest_frame.copy()
            
            # Check if grayscale and convert to 3 channels if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            # Convert OpenCV image to QImage
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Display image
            self.video_label.setPixmap(pixmap)
            
    def closeEvent(self, event):
        """Handle the close event."""
        # Stop timer
        self.update_timer.stop()
        event.accept() 