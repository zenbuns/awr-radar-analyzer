"""
Processing package for radar point cloud data.

This package contains modules for processing radar point cloud data,
including filtering, heatmap generation, and multi-frame processing.
"""

from .data_processor import (
    calculate_heatmap_size,
    filter_points_in_circle,
    update_heatmap_vectorized,
    update_live_heatmap_vectorized,
    apply_live_heatmap_decay,
    compute_heatmap_metrics
)

from .multi_frame import (
    process_multi_frame_data,
    combine_multi_frames,
    compute_multi_frame_metrics,
    load_latest_multi_frame_metrics
)

__all__ = [
    'calculate_heatmap_size',
    'filter_points_in_circle',
    'update_heatmap_vectorized',
    'update_live_heatmap_vectorized',
    'apply_live_heatmap_decay',
    'compute_heatmap_metrics',
    'process_multi_frame_data',
    'combine_multi_frames',
    'compute_multi_frame_metrics',
    'load_latest_multi_frame_metrics'
]
