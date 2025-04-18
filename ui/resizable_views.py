#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resizable window view for scatter plot and camera feed.

This module provides a simple integration to display both the scatter plot
and camera feed from ROS2 topic in separate resizable windows using PyQt5.
"""

import cv2
import numpy as np
import threading
import time
from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QWidget, QLabel, 
                            QMainWindow, QApplication, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Check for ROS2 availability
ROS2_AVAILABLE = False
try:
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available. Camera window will show test pattern.")


class ResizableImageWindow(QMainWindow):
    """
    A resizable PyQt window for displaying images.
    
    This class provides a simple window with a QLabel for displaying images
    that can be resized by the user.
    """
    
    def __init__(self, title, size=(640, 480)):
        """
        Initialize the window.
        
        Args:
            title (str): Window title.
            size (tuple): Initial window size (width, height).
        """
        super().__init__()
        
        # Set window properties
        self.setWindowTitle(title)
        self.resize(size[0], size[1])
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setMinimumSize(320, 240)
        layout.addWidget(self.image_label)
        
        # Initialize with black image
        self.update_image(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    def update_image(self, cv_image):
        """
        Update the displayed image.
        
        Args:
            cv_image (ndarray): OpenCV image (BGR format).
        """
        if cv_image is None:
            return
        
        # Convert OpenCV image to QImage
        height, width, channels = cv_image.shape
        bytes_per_line = channels * width
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create QPixmap and scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Set pixmap to label
        self.image_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        """Handle window close event."""
        event.accept()


class PyQtWindowManager:
    """
    Manages PyQt windows for displaying scatter plot and camera feed.
    
    This class handles the creation, updating, and destruction of PyQt windows
    for displaying both the scatter plot and camera feed from ROS2 topic.
    
    Attributes:
        windows (dict): Dictionary of active windows.
    """
    
    def __init__(self):
        """Initialize the PyQt window manager."""
        self.windows = {}
    
    def create_window(self, name, size=(640, 480)):
        """
        Create a new PyQt window.
        
        Args:
            name (str): Window name.
            size (tuple): Window size (width, height).
        """
        if name in self.windows:
            return
        
        # Create window
        window = ResizableImageWindow(name, size)
        window.show()
        
        self.windows[name] = window
    
    def update_window(self, name, image):
        """
        Update window with new image.
        
        Args:
            name (str): Window name.
            image (ndarray): Image to display.
        """
        if name not in self.windows:
            return
        
        self.windows[name].update_image(image)
    
    def close_window(self, name):
        """
        Close a PyQt window.
        
        Args:
            name (str): Window name.
        """
        if name in self.windows:
            self.windows[name].close()
            self.windows[name].deleteLater()
            del self.windows[name]
    
    def close_all_windows(self):
        """Close all PyQt windows."""
        window_names = list(self.windows.keys())
        for name in window_names:
            self.close_window(name)


class SimpleCameraSubscriber:
    """
    Simple subscriber for camera images from ROS2 topic.
    
    This class subscribes to a ROS2 Image topic and converts the images
    to OpenCV format for display.
    
    Attributes:
        node (Node): ROS2 node.
        subscription (Subscription): ROS2 subscription.
        latest_frame (ndarray): Latest received image.
        frame_received (bool): Flag indicating if a frame has been received.
    """
    
    def __init__(self, analyzer_node=None, topic='/out'):
        """
        Initialize the camera subscriber.
        
        Args:
            analyzer_node (Node): ROS2 node for subscription.
            topic (str): ROS2 topic to subscribe to.
        """
        self.analyzer_node = analyzer_node
        self.topic = topic
        self.subscription = None
        self.latest_frame = None
        self.frame_received = False
        
        if ROS2_AVAILABLE and self.analyzer_node:
            self.init_subscription()
        else:
            # Create test pattern if ROS2 is not available
            self.create_test_pattern()
    
    def init_subscription(self):
        """Initialize ROS2 subscription."""
        if not ROS2_AVAILABLE or not self.analyzer_node:
            return
        
        try:
            self.subscription = self.analyzer_node.create_subscription(
                Image,
                self.topic,
                self.image_callback,
                10  # QoS profile depth
            )
            print(f"Subscribed to ROS2 topic: {self.topic}")
        except Exception as e:
            print(f"Error creating ROS2 subscription: {e}")
            self.create_test_pattern()
    
    def create_test_pattern(self):
        """Create a test pattern image when no real camera feed is available."""
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
        
        self.frame_received = True
    
    def image_callback(self, msg):
        """Callback for when a new image is received."""
        if not ROS2_AVAILABLE:
            return
        
        try:
            # Extract image data
            height = msg.height
            width = msg.width
            encoding = msg.encoding
            
            # Convert to OpenCV image
            if encoding in ["bgr8", "rgb8"]:
                channels = 3
                data = np.frombuffer(msg.data, dtype=np.uint8)
                
                if encoding == "rgb8":
                    # RGB to BGR conversion
                    image = data.reshape((height, width, channels))
                    cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    # Already in BGR format
                    cv_image = data.reshape((height, width, channels))
                
                self.latest_frame = cv_image
                self.frame_received = True
            
            elif encoding == "mono8":
                # Grayscale image
                data = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = data.reshape((height, width))
                # Convert to BGR for display consistency
                self.latest_frame = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                self.frame_received = True
            
            else:
                print(f"Unsupported encoding: {encoding}")
        
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def get_latest_frame(self):
        """
        Get the latest received frame.
        
        Returns:
            The latest frame or None if no frame has been received.
        """
        if self.frame_received and self.latest_frame is not None:
            return self.latest_frame.copy()
        return None


class ScatterImageExporter:
    """
    Exports scatter plot images for display in PyQt windows.
    
    This class exports scatter plot images from the matplotlib figure
    to an OpenCV-compatible image format.
    """
    
    def __init__(self, scatter_view=None):
        """
        Initialize the scatter image exporter.
        
        Args:
            scatter_view (ScatterView): ScatterView instance to export from.
        """
        self.scatter_view = scatter_view
        self.latest_image = None
    
    def update_image(self):
        """
        Update the exported image from the scatter view.
        
        Returns:
            The exported image or None if not available.
        """
        if self.scatter_view is None or not hasattr(self.scatter_view, 'figure'):
            return None
        
        try:
            # Draw canvas to update figure
            self.scatter_view.canvas.draw()
            
            # Get the RGBA buffer from the figure
            w, h = self.scatter_view.figure.canvas.get_width_height()
            buf = np.frombuffer(self.scatter_view.figure.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (h, w, 3)
            
            # Convert RGB to BGR for OpenCV
            self.latest_image = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            return self.latest_image
        
        except Exception as e:
            print(f"Error exporting scatter image: {e}")
            return None
    
    def get_latest_image(self):
        """
        Get the latest exported image.
        
        Returns:
            The latest exported image or None if not available.
        """
        return self.latest_image


class WindowControlWidget(QWidget):
    """
    Widget for controlling resizable windows.
    
    This widget provides buttons for opening and closing resizable
    PyQt windows for scatter plot and camera feed visualization.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the window control widget.
        
        Args:
            parent: Parent widget (optional).
        """
        super().__init__(parent)
        
        # Get references to parent components
        self.main_window = parent
        self.analyzer = None
        self.scatter_view = None
        
        if hasattr(parent, 'analyzer'):
            self.analyzer = parent.analyzer
        
        if hasattr(parent, 'scatter_view'):
            self.scatter_view = parent.scatter_view
        
        # Create components
        self.window_manager = PyQtWindowManager()
        self.camera_subscriber = SimpleCameraSubscriber(self.analyzer)
        self.scatter_exporter = ScatterImageExporter(self.scatter_view)
        
        # Set up UI
        self.setup_ui()
        
        # Start timer for updating windows
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_windows)
        self.update_timer.start(33)  # ~30 fps
    
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Instructions label
        instructions = QLabel("Open resizable windows for scatter plot and camera feed:")
        instructions.setStyleSheet("font-weight: bold; color: white;")
        layout.addWidget(instructions)
        
        # Button to open/close scatter window
        self.scatter_button = QPushButton("Open Scatter Plot Window")
        self.scatter_button.setCheckable(True)
        self.scatter_button.toggled.connect(self.toggle_scatter_window)
        layout.addWidget(self.scatter_button)
        
        # Button to open/close camera window
        self.camera_button = QPushButton("Open Camera Window")
        self.camera_button.setCheckable(True)
        self.camera_button.toggled.connect(self.toggle_camera_window)
        layout.addWidget(self.camera_button)
        
        # Add spacer
        layout.addStretch()
    
    @pyqtSlot(bool)
    def toggle_scatter_window(self, checked):
        """
        Toggle scatter plot window visibility.
        
        Args:
            checked: Whether the window should be visible.
        """
        if checked:
            self.window_manager.create_window("Radar Scatter Plot", (800, 600))
            self.scatter_button.setText("Close Scatter Plot Window")
        else:
            self.window_manager.close_window("Radar Scatter Plot")
            self.scatter_button.setText("Open Scatter Plot Window")
    
    @pyqtSlot(bool)
    def toggle_camera_window(self, checked):
        """
        Toggle camera window visibility.
        
        Args:
            checked: Whether the window should be visible.
        """
        if checked:
            self.window_manager.create_window("Camera Feed", (640, 480))
            self.camera_button.setText("Close Camera Window")
        else:
            self.window_manager.close_window("Camera Feed")
            self.camera_button.setText("Open Camera Window")
    
    @pyqtSlot()
    def update_windows(self):
        """Update the content of all open windows."""
        # Update scatter plot window
        if "Radar Scatter Plot" in self.window_manager.windows:
            scatter_image = self.scatter_exporter.update_image()
            if scatter_image is not None:
                self.window_manager.update_window("Radar Scatter Plot", scatter_image)
        
        # Update camera window
        if "Camera Feed" in self.window_manager.windows:
            camera_frame = self.camera_subscriber.get_latest_frame()
            if camera_frame is not None:
                self.window_manager.update_window("Camera Feed", camera_frame)
    
    def closeEvent(self, event):
        """Handle the close event."""
        self.update_timer.stop()
        self.window_manager.close_all_windows()
        event.accept() 