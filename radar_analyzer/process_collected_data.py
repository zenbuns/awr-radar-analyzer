#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate distance band calculation fixes.
This script simulates radar data points at specific distances and verifies
that the distance band counting is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class MockParams:
    """Mock parameters class to simulate the RadarExperimentParams"""
    max_range: float = 30.0
    circle_interval: float = 10.0
    target_distance: float = 5.0
    use_recent_frames_only: bool = False
    
@dataclass
class MockLogger:
    """Mock logger to capture logging output"""
    logs: List[str] = field(default_factory=list)
    
    def info(self, msg: str) -> None:
        """Log info message"""
        print(f"INFO: {msg}")
        self.logs.append(f"INFO: {msg}")
        
    def warn(self, msg: str) -> None:
        """Log warning message"""
        print(f"WARN: {msg}")
        self.logs.append(f"WARN: {msg}")
        
    def error(self, msg: str) -> None:
        """Log error message"""
        print(f"ERROR: {msg}")
        self.logs.append(f"ERROR: {msg}")
        
@dataclass
class MockAnalyzer:
    """Mock analyzer class to simulate the RadarPointCloudAnalyzer"""
    params: MockParams = field(default_factory=MockParams)
    logger: MockLogger = field(default_factory=MockLogger)
    config_results: Dict[str, Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    
    def get_logger(self):
        """Return the logger"""
        return self.logger


def generate_points_at_distances(distances, points_per_distance=10, angle_spread=180):
    """
    Generate radar points at specific distances from the origin.
    
    Args:
        distances: List of target distances for points
        points_per_distance: Number of points to generate at each distance
        angle_spread: Angle spread in degrees for point distribution
        
    Returns:
        Tuple of (x_array, y_array, intensities_array)
    """
    x_points = []
    y_points = []
    intensities = []
    
    for distance in distances:
        for _ in range(points_per_distance):
            # Generate random angle within the specified spread
            # For a radar in front, we use angles around 90 degrees
            # (assuming 0 is to the right, 90 is straight ahead)
            center_angle = 90  # degrees (straight ahead)
            half_spread = angle_spread / 2
            angle_deg = np.random.uniform(center_angle - half_spread, center_angle + half_spread)
            angle_rad = np.radians(angle_deg)
            
            # Add small random variation to distance (±5%)
            actual_distance = distance * np.random.uniform(0.95, 1.05)
            
            # Calculate x, y coordinates
            x = actual_distance * np.cos(angle_rad)
            y = actual_distance * np.sin(angle_rad)
            
            # Generate random intensity (higher for closer points)
            intensity = 20 - (actual_distance * 0.5) + np.random.normal(0, 2)
            intensity = max(1, min(20, intensity))  # Clamp between 1-20
            
            x_points.append(x)
            y_points.append(y)
            intensities.append(intensity)
    
    return np.array(x_points), np.array(y_points), np.array(intensities)


def analyze_distance_bands(x_array, y_array, intensities_array, analyzer):
    """
    Analyze distance bands with the fixed calculation logic.
    
    Args:
        x_array: X-coordinates of points
        y_array: Y-coordinates of points
        intensities_array: Intensity values of points
        analyzer: MockAnalyzer instance
        
    Returns:
        Dictionary with distance band results
    """
    # Calculate distances from origin
    # For radar, forward might be y-axis positive, we need to consider direction
    # Consider the primary axis for distance measurement (usually forward direction)
    
    # Standard Euclidean distance
    distances_euclidean = np.sqrt(x_array ** 2 + y_array ** 2)
    
    # Alternative: distance along primary axis (Y-axis for radar typically points forward)
    # Use absolute values to ensure we're measuring distance regardless of direction
    distances_forward = np.abs(y_array)
    
    # By default use Euclidean, but we can switch based on a parameter
    use_directional = getattr(analyzer.params, 'use_directional_distance', False)
    
    if use_directional:
        # Use directional distance (along forward axis)
        distances = distances_forward
        analyzer.get_logger().info("Using directional (Y-axis) distance calculation")
    else:
        # Use standard Euclidean distance
        distances = distances_euclidean
        analyzer.get_logger().info("Using Euclidean distance calculation")
    
    # Create distance bins
    bins = np.arange(0, analyzer.params.max_range + analyzer.params.circle_interval, 
                    analyzer.params.circle_interval)
    counts, _ = np.histogram(distances, bins=bins)

    distance_bands = {}
    for i in range(len(counts)):
        band_key = f"{bins[i]}-{bins[i+1]}m"
        band_mask = (distances >= bins[i]) & (distances < bins[i+1])
        band_intensities = intensities_array[band_mask]
        band_count = float(np.sum(band_mask))  # More reliable count calculation
        avg_intensity = float(np.mean(band_intensities)) if band_intensities.size > 0 else 0.0
        
        # Store coordinates of points in this band for visual verification
        band_x = x_array[band_mask]
        band_y = y_array[band_mask]
        band_euclidean = distances_euclidean[band_mask]  # For comparison
        
        distance_bands[band_key] = {
            'count': band_count, 
            'avg_intensity': avg_intensity,
            'x_coordinates': band_x.tolist() if len(band_x) < 50 else "too many to list",
            'y_coordinates': band_y.tolist() if len(band_y) < 50 else "too many to list",
            'euclidean_distances': band_euclidean.tolist() if len(band_euclidean) < 50 else "too many to list"
        }

    # Find the band containing the target distance
    target_band_key = None
    target_distance = analyzer.params.target_distance
    
    # First try exact matching using the target distance
    for band_key in distance_bands.keys():
        band_start, band_end = band_key.split('-')
        band_start = float(band_start)
        band_end = float(band_end.replace('m', ''))
        
        if band_start <= target_distance < band_end:
            target_band_key = band_key
            analyzer.get_logger().info(f"Target band for {target_distance}m identified as {target_band_key}")
            break
    
    # Fallback to the first band if target band not found
    if target_band_key is None:
        target_band_key = list(distance_bands.keys())[0] if distance_bands else "0-0m"
        analyzer.get_logger().warn(f"Could not find exact band for {target_distance}m, using {target_band_key}")

    # Double-check the target band points with direct calculation
    target_band_start, target_band_end = target_band_key.split('-')
    target_band_start = float(target_band_start)
    target_band_end = float(target_band_end.replace('m', ''))
    
    # Create mask for target band points and count them directly
    target_band_mask = (distances >= target_band_start) & (distances < target_band_end)
    target_band_points = float(np.sum(target_band_mask))
    
    # Log detailed diagnostic information about target band orientation
    if np.sum(target_band_mask) > 0:
        target_x = x_array[target_band_mask]
        target_y = y_array[target_band_mask]
        min_x, max_x = np.min(target_x), np.max(target_x)
        min_y, max_y = np.min(target_y), np.max(target_y)
        analyzer.get_logger().info(f"Target band points X range: {min_x:.2f} to {max_x:.2f}")
        analyzer.get_logger().info(f"Target band points Y range: {min_y:.2f} to {max_y:.2f}")
    
    analyzer.get_logger().info(f"Points in distance range {target_band_start}-{target_band_end}m: {target_band_points}")
    analyzer.get_logger().info(f"Total points analyzed: {len(distances)}")
    
    # Special handling for specific bands for easier diagnostics
    for band_key, band_data in distance_bands.items():
        band_start, band_end = band_key.split('-')
        band_start = float(band_start)
        band_end = float(band_end.replace('m', ''))
        band_mask = (distances >= band_start) & (distances < band_end)
        actual_count = float(np.sum(band_mask))
        
        if abs(actual_count - band_data['count']) > 0.01:
            analyzer.get_logger().warn(f"Count mismatch in band {band_key}: stored={band_data['count']}, actual={actual_count}")
            # Update to the correct count
            distance_bands[band_key]['count'] = actual_count
    
    # Create results dictionary
    results = {
        'distance_bands': distance_bands,
        'target_band': target_band_key,
        'target_band_points': target_band_points,
        'total_points': len(x_array),
        'using_directional_distance': use_directional
    }
    
    return results


def plot_points_with_bands(x_array, y_array, distances, results):
    """
    Plot the points colored by distance band for visualization.
    
    Args:
        x_array: X-coordinates of points
        y_array: Y-coordinates of points
        distances: Array of distances from origin for each point
        results: Results dictionary from analyze_distance_bands
    """
    plt.figure(figsize=(10, 8))
    
    # Set up colormap for different distance bands
    distance_bands = results['distance_bands']
    band_colors = {}
    cmap = plt.cm.viridis
    
    for i, band_key in enumerate(sorted(distance_bands.keys())):
        band_colors[band_key] = cmap(i / len(distance_bands))
    
    # Plot each band with a different color
    for band_key, band_data in sorted(distance_bands.items()):
        band_start, band_end = band_key.split('-')
        band_start = float(band_start)
        band_end = float(band_end.replace('m', ''))
        
        # Create mask for this band
        band_mask = (distances >= band_start) & (distances < band_end)
        band_x = x_array[band_mask]
        band_y = y_array[band_mask]
        
        if len(band_x) > 0:
            plt.scatter(band_x, band_y, c=[band_colors[band_key]], 
                       label=f"{band_key}: {band_data['count']:.0f} points", 
                       alpha=0.7, s=50)
    
    # Highlight target band
    target_band = results['target_band']
    target_band_start, target_band_end = target_band.split('-')
    target_band_start = float(target_band_start)
    target_band_end = float(target_band_end.replace('m', ''))
    
    # Draw band boundaries as circles
    for r in np.arange(0, 31, 10):
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.text(0, r, f"{r}m", ha='center', va='bottom', color='gray')
    
    # Draw target distance circle
    target_distance = 5.0
    target_circle = plt.Circle((0, 0), target_distance, fill=False, linestyle='-', color='red', alpha=0.7)
    plt.gca().add_patch(target_circle)
    plt.text(0, target_distance, f"Target: {target_distance}m", ha='center', va='bottom', color='red')
    
    # Set up plot
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title(f"Radar Points by Distance Band\nTotal: {results['total_points']} points, Target Band: {results['target_band_points']:.0f} points")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis('equal')
    plt.legend(loc='upper right')
    
    # Save the plot
    plt.savefig("distance_band_test.png")
    plt.close()


def main():
    """Main test function"""
    print("=== Testing Distance Band Calculation Fix ===")
    
    # Create mock analyzer
    analyzer = MockAnalyzer()
    analyzer.params.target_distance = 5.0
    
    # Create test scenarios
    print("\n=== Test 1: Points at exact target distance (5m) ===")
    x, y, intensities = generate_points_at_distances([5.0], points_per_distance=20)
    results1 = analyze_distance_bands(x, y, intensities, analyzer)
    
    print("\n=== Test 2: Points spread across multiple distances ===")
    # Generate points at various distances
    distances = [1.0, 3.0, 5.0, 7.0, 9.0, 12.0, 15.0, 25.0]
    points_per_dist = [5, 10, 20, 15, 10, 5, 3, 2]  # More points around target distance
    
    all_x = []
    all_y = []
    all_intensities = []
    
    for dist, count in zip(distances, points_per_dist):
        x, y, i = generate_points_at_distances([dist], points_per_distance=count)
        all_x.extend(x)
        all_y.extend(y)
        all_intensities.extend(i)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_intensities = np.array(all_intensities)
    
    # Analyze with our fixed method
    results2 = analyze_distance_bands(all_x, all_y, all_intensities, analyzer)
    
    # Plot the results
    all_distances = np.sqrt(all_x**2 + all_y**2)
    plot_points_with_bands(all_x, all_y, all_distances, results2)
    
    # Print summary
    print("\n=== Results Summary ===")
    print(f"Test 1 - Points at target distance (5m):")
    print(f"  Total points: {results1['total_points']}")
    print(f"  Target band: {results1['target_band']}")
    print(f"  Points in target band: {results1['target_band_points']}")
    
    print(f"\nTest 2 - Points across multiple distances:")
    print(f"  Total points: {results2['total_points']}")
    print(f"  Target band: {results2['target_band']}")
    print(f"  Points in target band: {results2['target_band_points']}")
    
    print("\nDistance bands:")
    for band, data in sorted(results2['distance_bands'].items()):
        print(f"  {band}: {data['count']:.0f} points")
    
    print("\n=== Verification ===")
    # Verify total points matches sum of bands
    band_sum = sum(data['count'] for data in results2['distance_bands'].values())
    print(f"Sum of band points: {band_sum:.0f}")
    print(f"Total points: {results2['total_points']}")
    
    if abs(band_sum - results2['total_points']) < 0.01:
        print("✅ Verification passed: Band sum matches total points")
    else:
        print("❌ Verification failed: Band sum does not match total points")
    
    # Verify target band points is correct
    target_band = results2['target_band']
    expected_target_points = results2['distance_bands'][target_band]['count']
    actual_target_points = results2['target_band_points']
    
    if abs(expected_target_points - actual_target_points) < 0.01:
        print("✅ Verification passed: Target band points calculation is correct")
    else:
        print(f"❌ Verification failed: Target band points mismatch: {expected_target_points} vs {actual_target_points}")
    
    print("\nA visualization of the points by distance band has been saved to 'distance_band_test.png'")


if __name__ == "__main__":
    main() 