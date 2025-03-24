#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Matplotlib toolbar with non-blocking save functionality.

This module provides a custom implementation of the Matplotlib NavigationToolbar
that replaces the built-in save functionality with a non-blocking approach
that won't freeze the UI.
"""

import os
import datetime
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QApplication, 
                            QProgressDialog, QPushButton)
from PyQt5.QtCore import Qt, QTimer


class NonBlockingNavigationToolbar(NavigationToolbar2QT):
    """
    Custom Matplotlib toolbar that prevents UI freezing during save operations.
    
    This class overrides the default save behavior to use a non-blocking
    QFileDialog approach that keeps the UI responsive.
    """
    
    def __init__(self, canvas, parent, coordinates=True, view_type="heatmap"):
        """
        Initialize the custom toolbar.
        
        Args:
            canvas: The Matplotlib canvas.
            parent: Parent widget.
            coordinates: Whether to show coordinates.
            view_type: The type of view ("heatmap" or "scatter") to customize the save behavior.
        """
        super().__init__(canvas, parent, coordinates)
        self.view_type = view_type
        self.parent = parent
        self.save_directory = os.path.join(os.path.expanduser("~"), "radar_experiment_data", view_type + "s")
        
        # Create save directory if it doesn't exist
        try:
            os.makedirs(self.save_directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating save directory: {e}")
            self.save_directory = os.path.expanduser("~")
    
    def save_figure(self, *args):
        """
        Override the default save_figure method with a non-blocking version.
        
        This method replaces the built-in Matplotlib save dialog with a non-blocking
        QFileDialog to prevent UI freezing.
        """
        # Generate a default filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"radar_{self.view_type}_{timestamp}.png"
        default_path = os.path.join(self.save_directory, default_filename)
        
        # Create a non-blocking QFileDialog
        dialog = QFileDialog(self.parent, f"Save {self.view_type.capitalize()} Figure", default_path)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("PNG Image (*.png);;PDF Document (*.pdf);;SVG Image (*.svg);;All Files (*)")
        dialog.setDefaultSuffix("png")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)  # Use Qt's dialog for better control
        dialog.setWindowModality(Qt.WindowModal)  # Make dialog modal to parent only
        
        # Process events to keep UI responsive
        QApplication.processEvents()
        
        # Show dialog and wait for result
        if dialog.exec_() == QFileDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files and selected_files[0]:
                file_path = selected_files[0]
                
                # Remember the directory for next time
                self.save_directory = os.path.dirname(file_path)
                
                # Create a progress dialog to show saving is in progress
                progress = QProgressDialog(f"Saving {self.view_type}...", "Cancel", 0, 100, self.parent)
                progress.setWindowTitle(f"Saving {self.view_type.capitalize()}")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(500)  # Only show for operations taking longer than 500ms
                progress.setValue(10)
                QApplication.processEvents()
                
                try:
                    # Update progress
                    progress.setValue(30)
                    QApplication.processEvents()
                    
                    # Save the figure without blocking UI
                    dpi = 300  # Higher DPI for good quality prints
                    self.canvas.figure.savefig(file_path, dpi=dpi, bbox_inches='tight')
                    
                    # Update progress
                    progress.setValue(70)
                    QApplication.processEvents()
                    
                    # Also save the raw data for heatmap views
                    data_saved = False
                    if self.view_type == "heatmap" and hasattr(self.parent, "heatmap_data") and self.parent.heatmap_data is not None:
                        data_path = os.path.splitext(file_path)[0] + '.npz'
                        np.savez_compressed(data_path, data=self.parent.heatmap_data)
                        data_saved = True
                    # For scatter views, save point cloud data if available
                    elif self.view_type == "scatter" and hasattr(self.parent, "latest_points"):
                        if hasattr(self.parent, "latest_x") and hasattr(self.parent, "latest_y") and hasattr(self.parent, "latest_intensities"):
                            data_path = os.path.splitext(file_path)[0] + '.npz'
                            np.savez_compressed(data_path, 
                                              x=self.parent.latest_x, 
                                              y=self.parent.latest_y, 
                                              intensities=self.parent.latest_intensities)
                            data_saved = True
                    
                    # Complete progress
                    progress.setValue(100)
                    QApplication.processEvents()
                    
                    # Close progress dialog after a short delay
                    QTimer.singleShot(500, progress.close)
                    
                    # Show success message with details about what was saved
                    data_message = "\nRaw data was also saved." if data_saved else ""
                    QMessageBox.information(self.parent, "Success", 
                                           f"{self.view_type.capitalize()} image saved to:\n{file_path}{data_message}")
                    
                except Exception as e:
                    # Close progress in case of error
                    progress.close()
                    
                    QMessageBox.critical(self.parent, "Error", 
                                        f"Failed to save {self.view_type}: {str(e)}")
                    print(f"Error saving {self.view_type}: {str(e)}")
        
        # Make sure to process events after dialog closes
        QApplication.processEvents() 