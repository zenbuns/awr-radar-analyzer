    def set_colormap(self, colormap_name):
        """
        Set the colormap for visualizations.
        
        Args:
            colormap_name: Name of the matplotlib colormap to use.
        """
        self.colormap = colormap_name
        # Update any visualization settings that depend on colormap
        if hasattr(self, 'update_visualization_settings'):
            self.update_visualization_settings() 