"""
Utility modules for radar data processing and visualization.

This package contains modules for processing radar data and creating
visualizations for scientific analysis.
"""

from utils.data_processing import (
    calculate_snr,
    filter_points_by_distance,
    calculate_distance_bands,
    find_target_band,
    grid_heatmap_data,
    compute_circle_statistics,
    apply_gaussian_smoothing,
    analyze_heatmap_region
)

from utils.visualization import (
    setup_radar_scatter_figure,
    setup_heatmap_figure,
    add_contours_to_heatmap,
    save_scientific_visualization,
    update_statistics_text
)

__all__ = [
    'calculate_snr',
    'filter_points_by_distance',
    'calculate_distance_bands',
    'find_target_band',
    'grid_heatmap_data',
    'compute_circle_statistics',
    'apply_gaussian_smoothing',
    'analyze_heatmap_region',
    'setup_radar_scatter_figure',
    'setup_heatmap_figure',
    'add_contours_to_heatmap',
    'save_scientific_visualization',
    'update_statistics_text'
] 