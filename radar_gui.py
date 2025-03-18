#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt-based GUI for radar point cloud analysis.

This module provides the main GUI application for the radar point cloud analyzer,
integrating the ROS node with the PyQt user interface.
"""

import sys
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from ui.main_window import MainWindow
from radar_analyzer.core import RadarPointCloudAnalyzer


class RadarAnalyzerGUI:
    """
    Main GUI application for radar point cloud analysis.
    
    This class initializes the ROS node, sets up the PyQt application,
    and handles the communication between them.
    
    Attributes:
        analyzer_node: The ROS node for radar analysis.
        app: The PyQt application instance.
        main_window: The main application window.
    """
    
    def __init__(self):
        """Initialize the radar analyzer GUI application."""
        self.analyzer_node = None
        self.app = None
        self.main_window = None
    
    def initialize_ros(self, args=None):
        """
        Initialize the ROS node.
        
        Args:
            args: Command line arguments for ROS initialization.
            
        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            
            # Initialize ROS
            rclpy.init(args=args)
            
            # Create analyzer node
            self.analyzer_node = RadarPointCloudAnalyzer()
            
            # Create executor for ROS
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.analyzer_node)
            
            # Start ROS spin in separate thread
            self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.ros_thread.start()
            
            return True
        
        except ImportError:
            print("ROS 2 (rclpy) not available. Running in visualization-only mode.")
            return False
        
        except Exception as e:
            print(f"Error initializing ROS: {str(e)}")
            if 'rclpy' in str(e).lower():
                print("This appears to be a ROS 2 error. Make sure ROS 2 is properly installed and sourced.")
            return False
    
    def initialize_gui(self):
        """
        Initialize the PyQt GUI application.
        
        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            # Create PyQt application
            self.app = QApplication(sys.argv)
            
            # Enable high DPI scaling
            self.app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            self.app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            
            # Set application style
            self.app.setStyle('Fusion')
            
            # Create main window with analyzer
            self.main_window = MainWindow(self.analyzer_node)
            
            # Show window
            self.main_window.show()
            
            return True
        
        except Exception as e:
            print(f"Error initializing GUI: {str(e)}")
            if 'PyQt5' in str(e).lower():
                print("This appears to be a PyQt error. Make sure PyQt5 is properly installed.")
            return False
    
    def run(self):
        """
        Run the application.
        
        This method starts the PyQt event loop and handles cleanup
        when the application exits.
        
        Returns:
            The application exit code.
        """
        exit_code = self.app.exec_()
        
        # Cleanup
        print("Shutting down...")
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown()
            if self.analyzer_node:
                self.analyzer_node.destroy_node()
            import rclpy
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        
        return exit_code


def main(args=None):
    """
    Main entry point for the application.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Application exit code.
    """
    # Create GUI application
    gui = RadarAnalyzerGUI()
    
    # Initialize ROS (optional, will run in viz-only mode if it fails)
    gui.initialize_ros(args)
    
    # Initialize GUI
    if gui.initialize_gui():
        # Run application
        return gui.run()
    else:
        print("Failed to initialize application.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
