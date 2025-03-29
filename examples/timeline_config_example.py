"""
Timeline Configuration Example

This example demonstrates how to use timeline configuration in the VideoAI project.
It shows how to access timeline configuration, create timelines with default settings,
and override specific settings.
"""

from pathlib import Path
import sys
import os

# Add parent directory to import path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import VideoAI components
from config import get_timeline_config, config
from timeline_manager import TimelineManager
from file_manager import FileManager

def timeline_config_example():
    """
    Demonstrate timeline configuration features.
    """
    # Print the global timeline configuration
    print("\n=== Global Timeline Configuration ===")
    global_config = get_timeline_config()
    print(f"Storage Directory: {global_config.storage_directory}")
    print(f"Default Resolution: {global_config.default_width}x{global_config.default_height}")
    print(f"Default Framerate: {global_config.default_framerate}")
    print(f"Default Samplerate: {global_config.default_samplerate}")
    
    print("\nVisualization Settings:")
    print(f"  Detail Level: {global_config.visualization.default_detail_level}")
    print(f"  Max Width: {global_config.visualization.max_width}")
    print(f"  Time Markers: {global_config.visualization.time_markers}")
    
    print("\nSerialization Settings:")
    print(f"  Format: {global_config.serialization.format}")
    print(f"  Compression: {global_config.serialization.compression}")
    print(f"  Auto Backup: {global_config.serialization.auto_backup}")
    
    # Print channel-specific timeline configuration (for Channel 1)
    print("\n=== Channel 1 Timeline Configuration ===")
    channel_config = get_timeline_config(channel_number=1)
    print(f"Storage Directory: {channel_config.storage_directory}")
    print(f"Default Resolution: {channel_config.default_width}x{channel_config.default_height}")
    print(f"Default Framerate: {channel_config.default_framerate}")
    print(f"Default Samplerate: {channel_config.default_samplerate}")
    
    print("\nVisualization Settings:")
    print(f"  Detail Level: {channel_config.visualization.default_detail_level}")
    print(f"  Max Width: {channel_config.visualization.max_width}")
    print(f"  Time Markers: {channel_config.visualization.time_markers}")
    
    # Create timeline managers with different configurations
    print("\n=== Creating Timeline Managers ===")
    global_tm = TimelineManager()
    channel_tm = TimelineManager(channel_number=1)
    
    print(f"Global TimelineManager Channel: {global_tm.channel_number}")
    print(f"Channel TimelineManager Channel: {channel_tm.channel_number}")
    
    # Get timeline paths for different configurations
    print("\n=== Timeline File Paths ===")
    file_mgr = FileManager()
    
    global_timeline_dir = file_mgr.get_timeline_directory()
    channel_timeline_dir = file_mgr.get_timeline_directory(channel_number=1)
    
    print(f"Global Timeline Directory: {global_timeline_dir}")
    print(f"Channel Timeline Directory: {channel_timeline_dir}")
    
    global_timeline_path = file_mgr.get_timeline_path("example_timeline")
    channel_timeline_path = file_mgr.get_timeline_path("example_timeline", channel_number=1)
    
    print(f"Global Timeline Path: {global_timeline_path}")
    print(f"Channel Timeline Path: {channel_timeline_path}")
    
    # Get visualization paths for different configurations
    print("\n=== Visualization File Paths ===")
    
    global_viz_dir = file_mgr.get_timeline_visualization_directory()
    channel_viz_dir = file_mgr.get_timeline_visualization_directory(channel_number=1)
    
    print(f"Global Visualization Directory: {global_viz_dir}")
    print(f"Channel Visualization Directory: {channel_viz_dir}")
    
    global_viz_path = file_mgr.get_timeline_visualization_path("example_timeline", detail_level="normal")
    channel_viz_path = file_mgr.get_timeline_visualization_path("example_timeline", detail_level="detailed", channel_number=1)
    
    print(f"Global Visualization Path: {global_viz_path}")
    print(f"Channel Visualization Path: {channel_viz_path}")
    
    # Demonstrate modifying configuration
    print("\n=== Modifying Configuration ===")
    print("Original timeline serialization format:", config.timeline.serialization.format)
    print("Original timeline visualization detail level:", config.timeline.visualization.default_detail_level)
    
    # Temporarily modify configuration for demonstration
    old_format = config.timeline.serialization.format
    old_detail_level = config.timeline.visualization.default_detail_level
    
    config.timeline.serialization.format = "json-compressed"
    config.timeline.visualization.default_detail_level = "detailed"
    
    print("Modified timeline serialization format:", config.timeline.serialization.format)
    print("Modified timeline visualization detail level:", config.timeline.visualization.default_detail_level)
    
    # Restore original configuration
    config.timeline.serialization.format = old_format
    config.timeline.visualization.default_detail_level = old_detail_level
    
    print("Restored timeline serialization format:", config.timeline.serialization.format)
    print("Restored timeline visualization detail level:", config.timeline.visualization.default_detail_level)

if __name__ == "__main__":
    timeline_config_example()