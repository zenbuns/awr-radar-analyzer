#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="radar_point_cloud_analyzer",
    version="1.0.0",
    description="A comprehensive tool for analyzing radar point clouds from an AWR1843 mmWave radar",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/radar_point_cloud_analyzer",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "PyQt5>=5.15.0",
    ],
    extras_require={
        "ros": ["rclpy>=1.0.0", "sensor_msgs_py>=0.2.0"],
        "dev": ["pytest>=6.0.0", "pylint>=2.5.0"],
    },
    entry_points={
        "console_scripts": [
            "radar_analyzer=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
