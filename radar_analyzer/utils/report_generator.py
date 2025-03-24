#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report generator utilities for radar data analysis.

This module contains functions for generating reports and comparisons
of radar data collection results.
"""

import os
from datetime import datetime
import numpy as np
from radar_analyzer.processing.multi_frame import load_latest_multi_frame_metrics


def generate_comparison_report(analyzer) -> str:
    """
    Generate an HTML comparison report for all collected configurations.
    
    This method creates a comprehensive HTML report comparing the results
    of different radar configurations tested during the experiment.

    Args:
        analyzer: RadarPointCloudAnalyzer instance.

    Returns:
        The path to the generated HTML report, or None if generation failed.
    """
    if not analyzer.config_results:
        analyzer.get_logger().warn("No data available for report generation")
        return None

    # Add debug logging to show available data
    analyzer.get_logger().info(f"Generating report with {len(analyzer.config_results)} configurations")
    for config, distances in analyzer.config_results.items():
        analyzer.get_logger().info(f"Config: {config} with {len(distances)} distances")
        for distance, results in distances.items():
            analyzer.get_logger().info(f"  Distance: {distance}m")
            
            # Log all top-level keys for debugging
            analyzer.get_logger().info(f"  Available keys: {', '.join(results.keys())}")
            
            # Log specific ROI-related keys if they exist
            roi_keys = [k for k in results.keys() if 'roi' in k.lower()]
            if roi_keys:
                analyzer.get_logger().info(f"  ROI-related keys: {', '.join(roi_keys)}")
            
            # Check if multi_frame_metrics exists and log its keys
            if 'multi_frame_metrics' in results and results['multi_frame_metrics']:
                mf_keys = results['multi_frame_metrics'].keys()
                analyzer.get_logger().info(f"  Multi-frame metrics keys: {', '.join(mf_keys)}")
                
                # Log ROI-specific metrics if they exist
                mf_roi_keys = [k for k in mf_keys if 'roi' in k.lower()]
                if mf_roi_keys:
                    analyzer.get_logger().info(f"  Multi-frame ROI keys: {', '.join(mf_roi_keys)}")

    # Load the latest multi-frame metrics from json files
    has_multi_frame_data = load_latest_multi_frame_metrics(analyzer)

    try:
        data_dir = os.path.expanduser('~/radar_experiment_data')
        os.makedirs(data_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(data_dir, f"comparison_report_{timestamp}.html")

        with open(report_file, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Radar Experiment Comparison Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .chart-container { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
                </style>
            </head>
            <body>
                <h1>Radar Experiment Comparison Report</h1>
                <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            """)

            f.write("""
                <h2>Configuration Summary</h2>
                <table>
                    <tr>
                        <th>Configuration</th>
                        <th>Target Distance</th>
                        <th>Total Points</th>
                        <th>Primary ROI</th>
                        <th>Outside ROI</th>
                        <th>Point Density</th>
                        <th>Avg. Intensity</th>
                    </tr>
            """)

            for config, distances in analyzer.config_results.items():
                for distance, results in distances.items():
                    points = results.get('circle_points', 0)
                    avg_intensity = results.get('circle_avg_intensity', 0)
                    roi_points = results.get('roi_combined_point_count', 0)
                    outside_roi_points = results.get('outside_roi_combined_point_count', 0)
                    circle_area = np.pi * analyzer.params.circle_radius**2
                    if circle_area == 0:
                        density = 0
                    else:
                        density = points / circle_area

                    f.write(f"""
                        <tr>
                            <td>{config}</td>
                            <td>{distance}m</td>
                            <td>{points}</td>
                            <td>{roi_points}</td>
                            <td>{outside_roi_points}</td>
                            <td>{density:.2f} pts/m²</td>
                            <td>{avg_intensity:.2f}</td>
                        </tr>
                    """)

            f.write("</table>")

            f.write("<h2>Detailed Results</h2>")

            for config, distances in analyzer.config_results.items():
                f.write(f"<h3>Configuration: {config}</h3>")

                for distance, results in distances.items():
                    # Get distance calculation method
                    distance_method = "Directional (Forward)" if results.get('using_directional_distance', False) else "Radial (Euclidean)"
                    
                    f.write(f"""
                        <div class="chart-container">
                            <h4>Target Distance: {distance}m</h4>
                            <p>Total Points: {results.get('total_points', 0)}</p>
                            <p>Average Intensity: {results.get('avg_intensity', 0):.2f}</p>

                            <h5>Distance Band Analysis (All {results.get('frames_analyzed', 'Available')} Frames)</h5>
                            <p><em>Distance calculation method: {distance_method}</em></p>
                            <table>
                                <tr>
                                    <th>Distance Band</th>
                                    <th>Points</th>
                                    <th>Percentage</th>
                                    <th>Avg. Intensity</th>
                                </tr>
                    """)

                    distance_bands = results.get('distance_bands', {})
                    target_band = results.get('target_band', '')
                    target_band_count = results.get('target_band_points', 0)
                    total_count = results.get('total_points', 0)
                    
                    # Check for metadata with more accurate distance band information
                    if 'metadata' in results:
                        metadata = results.get('metadata', {})
                        if 'distance_bands' in metadata:
                            # Use the more detailed metadata information
                            distance_bands = metadata.get('distance_bands', {})
                            target_band = metadata.get('target_band', target_band)
                            target_band_count = metadata.get('target_band_count', target_band_count)
                            total_count = metadata.get('total_count', total_count)
                    
                    for band, band_data in distance_bands.items():
                        highlight = ''
                        count = 0
                        avg_intensity = 0
                        
                        # Handle both dictionary and scalar values
                        if isinstance(band_data, dict):
                            count = band_data.get('count', 0)
                            avg_intensity = band_data.get('avg_intensity', 0)
                        else:
                            count = band_data  # If band_data is just the count
                            
                        percentage = (count / total_count * 100) if total_count > 0 else 0
                        
                        if band == target_band:
                            highlight = 'style="background-color: #ffffcc; font-weight: bold;"'
                            
                        f.write(f"""
                            <tr {highlight}>
                                <td>{band}</td>
                                <td>{count:.0f}</td>
                                <td>{percentage:.1f}%</td>
                                <td>{avg_intensity:.2f}</td>
                            </tr>
                        """)

                    f.write("</table></div>")
            
            # Add Multi-Frame Processing section if enabled
            # Ensure we always show multi-frame metrics when available
            if hasattr(analyzer, 'multi_frame_metrics') and analyzer.multi_frame_metrics:
                f.write("""
                <h2>Multi-Frame Processing Results</h2>
                <p>Analysis of 10-frame averaging for AWR1843BOOST radar</p>
                """)
                
                # Loop through each configuration with multi-frame metrics
                for config_name, metrics in analyzer.multi_frame_metrics.items():
                    f.write(f"""
                    <div class="chart-container">
                        <h3>Configuration: {config_name}</h3>
                        <p>Method: {analyzer.params.multi_frame_method}, Frames Combined: 10</p>
                        
                        <h4>ROI Analysis</h4>
                        <table>
                            <tr>
                                <th>ROI Type</th>
                                <th>Points</th>
                                <th>Avg Pts/Frame</th>
                                <th>Density</th>
                                <th>SNR</th>
                            </tr>
                            <tr>
                                <td>Primary ROI</td>
                                <td>{metrics.get('roi_combined_point_count', 0)}</td>
                                <td>{metrics.get('roi_avg_single_frame_count', 0):.1f}</td>
                                <td>{metrics.get('roi_spatial_density', 0):.2f}</td>
                                <td>{metrics.get('roi_snr_db', 0):.1f} dB</td>
                            </tr>
                            <tr>
                                <td>Outside ROI</td>
                                <td>{metrics.get('outside_roi_combined_point_count', 0)}</td>
                                <td>{metrics.get('outside_roi_avg_single_frame_count', 0):.1f}</td>
                                <td>{metrics.get('outside_roi_spatial_density', 0):.2f}</td>
                                <td>{metrics.get('outside_roi_snr_db', 0):.1f} dB</td>
                            </tr>
                        </table>
                        
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Description</th>
                            </tr>
                    """)
                    
                    # Add rows for each multi-frame metric with descriptions
                    metric_descriptions = {
                        'point_density_improvement': 'Improvement in point density compared to single frames',
                        'combined_point_count': 'Total points in combined frame',
                        'avg_single_frame_count': 'Average points per single frame',
                        'combined_avg_intensity': 'Average intensity in combined frame',
                        'combined_max_intensity': 'Maximum intensity in combined frame',
                        'combined_intensity_std': 'Standard deviation of intensities in combined frame',
                        'coverage_percentage': 'Percentage of voxels filled in combined frame',
                        'noise_reduction_factor': 'Noise reduction compared to single frames',
                        'snr_dB': 'Signal-to-noise ratio in dB for combined frame',
                        # New 10-frame average metrics
                        'ten_frame_avg_point_count': 'Average point count across 10 frames',
                        'ten_frame_avg_intensity': 'Average intensity across 10 frames',
                        'ten_frame_stability': 'Stability of point counts across 10 frames (1.0 is perfect)',
                        'ten_frame_snr_dB': 'Signal-to-noise ratio in dB across 10 frames'
                    }
                    
                    for metric, value in metrics.items():
                        description = metric_descriptions.get(metric, 'Additional metric')
                        formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                        f.write(f"""
                            <tr>
                                <td>{metric}</td>
                                <td><b>{formatted_value}</b></td>
                                <td>{description}</td>
                            </tr>
                        """)
                    
                    f.write("""
                        </table>
                        
                        <h4>Performance Summary</h4>
                        <div style="background-color: #f8f8f8; padding: 10px; border-radius: 5px;">
                            <p>Analysis of 10-frame averaging:</p>
                            <ul>
                    """)
                    
                    # Add performance summary points based on metrics
                    if 'point_density_improvement' in metrics:
                        improvement = metrics['point_density_improvement']
                        f.write(f"<li>Point density improved by <b>{improvement:.1f}x</b> compared to single frames</li>")
                    
                    if 'combined_point_count' in metrics and 'avg_single_frame_count' in metrics:
                        combined = metrics['combined_point_count']
                        avg = metrics['avg_single_frame_count']
                        f.write(f"<li>Combined frame contains <b>{combined}</b> points vs average of <b>{avg:.1f}</b> points in single frames</li>")
                    
                    if 'noise_reduction_factor' in metrics:
                        noise_reduction = metrics['noise_reduction_factor']
                        f.write(f"<li>Noise reduced by a factor of <b>{noise_reduction:.1f}</b></li>")
                    
                    if 'snr_dB' in metrics:
                        snr = metrics['snr_dB']
                        f.write(f"<li>Signal-to-Noise Ratio: <b>{snr:.1f} dB</b></li>")
                    
                    if 'coverage_percentage' in metrics:
                        coverage = metrics['coverage_percentage']
                        f.write(f"<li>Spatial coverage: <b>{coverage:.1f}%</b> of the sensing area</li>")
                    
                    # Add 10-frame specific metrics to performance summary if available
                    if 'ten_frame_avg_point_count' in metrics:
                        avg_points = metrics['ten_frame_avg_point_count']
                        f.write(f"<li>10-Frame average point count: <b>{avg_points:.1f}</b> points</li>")
                        
                    if 'ten_frame_avg_intensity' in metrics:
                        avg_intensity = metrics['ten_frame_avg_intensity']
                        f.write(f"<li>10-Frame average intensity: <b>{avg_intensity:.2f}</b></li>")
                        
                    if 'ten_frame_stability' in metrics:
                        stability = metrics['ten_frame_stability']
                        quality = "High" if stability > 0.8 else "Medium" if stability > 0.5 else "Low"
                        f.write(f"<li>Frame-to-frame stability: <b>{stability:.2f}</b> ({quality})</li>")
                        
                    if 'ten_frame_snr_dB' in metrics:
                        snr = metrics['ten_frame_snr_dB']
                        f.write(f"<li>10-Frame SNR: <b>{snr:.1f} dB</b></li>")
                    
                    f.write("""
                            </ul>
                            <p>Overall, the 10-frame averaging technique significantly improves the radar's point cloud quality.</p>
                        </div>
                    </div>
                    """)
                
                # End the multi-frame metrics section

            f.write("""
                <script>
                    // Optional JavaScript for interactivity
                </script>
            </body>
            </html>
            """)

        analyzer.get_logger().info(f"Generated comparison report: {report_file}")
        return report_file

    except Exception as e:
        analyzer.get_logger().error(f"Error generating report: {str(e)}")
        return None