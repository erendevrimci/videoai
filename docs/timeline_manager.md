# Timeline Manager

The `TimelineManager` provides a powerful way to create, manipulate, and serialize video timelines in the VideoAI project. It serves as a bridge between VideoAI's clip-based editing and auto-editor's advanced timeline capabilities.

## Overview

The timeline system enables:

- Creating structured representations of edit decisions
- Exporting timelines to professional editing formats
- Visualizing edit sequences for better understanding
- Separating edit decisions from rendering for faster iteration
- Channel-aware timeline management

## Getting Started

### Basic Usage

```python
from timeline_manager import TimelineManager

# Initialize the timeline manager
timeline_mgr = TimelineManager(channel_number=1)

# Create an empty timeline
timeline = timeline_mgr.create_v3_timeline(
    width=1080,
    height=1920,
    framerate=30
)

# Visualize the timeline
visualization = timeline_mgr.visualize_timeline(timeline)
print(visualization)

# Save timeline to a file
timeline_mgr.serialize_timeline(timeline, "path/to/timeline.json")
```

### Converting VideoAI Clip Sequences

You can convert VideoAI's clip sequences directly to timelines:

```python
# Define a clip sequence (from video_edit.py)
clip_sequence = [
    {
        "clip_name": "sample_clips/clip1.mp4",
        "start_time": 0,
        "duration": 5.0,
        "script_segment": "First segment of the script"
    },
    {
        "clip_name": "sample_clips/clip2.mp4",
        "start_time": 2.0,
        "duration": 6.0,
        "script_segment": "Second segment of the script"
    }
]

# Convert to timeline
timeline = timeline_mgr.clip_sequence_to_timeline(
    clip_sequence,
    output_width=1080,
    output_height=1920,
    framerate=30
)
```

## Timeline Types

The timeline system supports two main timeline types:

### v1 Timeline

A simple timeline based on chunks of a single source video:

```python
# Create a v1 timeline from a video file
timeline = timeline_mgr.create_v1_timeline(
    source_path="path/to/video.mp4"
)
```

This timeline type is useful for simple cuts of a single video.

### v3 Timeline

An advanced timeline supporting multiple tracks, sources, and media types:

```python
# Create a v3 timeline
timeline = timeline_mgr.create_v3_timeline(
    width=1080,
    height=1920,
    framerate=30,
    samplerate=48000
)

# Add clips to the timeline
# See examples/timeline_example.py for detailed examples
```

This timeline type enables complex editing with multiple sources.

## Timeline Serialization

Timelines can be serialized to and from JSON format with rich metadata:

```python
# Serialize to dictionary
timeline_dict = timeline_mgr.serialize_timeline(timeline)

# Save to file with description
timeline_mgr.serialize_timeline(
    timeline, 
    "path/to/timeline.json",
    description="Main project timeline with intro and outro"
)

# Load from file with validation
timeline = timeline_mgr.deserialize_timeline("path/to/timeline.json", validate=True)

# Convenience methods for channel-specific timelines
timeline_mgr.save_timeline(timeline, "project_timeline")  # Saves to channel directory
loaded_timeline = timeline_mgr.load_timeline("project_timeline")
```

### JSON Schema Validation

Timeline JSON files are validated against a schema to ensure compatibility:

```python
# Validate while serializing
timeline_mgr.serialize_timeline(timeline, path, validate=True)
```

### Resolution Handling

The timeline system supports multiple formats for resolution values:

- Tuple of integers: `(1920, 1080)`
- Tuple of floats: `(1280.0, 720.0)`
- List of integers or floats: `[1920, 1080]`
- Object with width/height: `{"width": 1920, "height": 1080}`

Resolution values are automatically converted between these formats during serialization and deserialization:

```python
# Tuples are automatically converted to lists for JSON serialization
timeline.res = (1920, 1080)  
timeline_dict = timeline_mgr.serialize_timeline(timeline)
# In JSON: "resolution": [1920, 1080]

# Lists are automatically converted to tuples during deserialization
timeline = timeline_mgr.deserialize_timeline(timeline_path)
# Access as tuple: timeline.res[0], timeline.res[1]
```

### Custom Data Types

The serialization system handles special data types:

- Path objects (converted to strings)
- Fraction objects (converted to "numerator/denominator" strings)
- FileInfo objects (converted to their path string)
- Tuples (converted to lists for JSON compatibility)
- Chunks objects (converted to structured representations)

### Timeline Format Conversion

You can convert between timeline formats:

```python
# Convert v1 timeline to v3
v3_timeline = timeline_mgr.convert_v1_to_v3(v1_timeline)

# Convert v3 timeline back to v1 (if compatible)
v1_timeline = timeline_mgr.convert_v3_to_v1(v3_timeline)
```

## Channel-Aware Operations

The `TimelineManager` is channel-aware, allowing it to organize timelines by channel:

```python
# Initialize for a specific channel
timeline_mgr = TimelineManager(channel_number=2)

# Save timeline - will be stored in the channel's directory
timeline_mgr.serialize_timeline(
    timeline, 
    timeline_mgr.get_timeline_path("my_timeline")
)

# Or use the convenience method
timeline_mgr.save_timeline(timeline, "my_timeline")
```

## Timeline Visualization

The `TimelineManager` provides ASCII visualizations of timelines:

```python
# Get a text representation of the timeline
visualization = timeline_mgr.visualize_timeline(timeline)
print(visualization)
```

Example output for a v3 timeline:
```
global
 timebase 30/1
 samplerate 48000
 res 1080x1920

video
 v0 [#:start 0 #:dur 90 #:off 0] [#:start 90 #:dur 90 #:off 30] 

audio
 a0 [#:start 0 #:dur 90 #:off 0] [#:start 90 #:dur 90 #:off 30] 
```

## Timeline Integration with video_edit.py

The `TimelineManager` is designed to integrate with VideoAI's existing video editing pipeline. The timeline can be created from the existing clip selection process and then used for rendering or exporting to professional formats.

See the examples directory for detailed usage examples.

## Title and Description Integration

The timeline system includes comprehensive integration with title and description metadata, which is preserved during serialization and deserialization. The title and description metadata is stored in the timeline's `videoai_metadata` attribute and can be shared separately with upload systems.

### Adding Metadata to Timelines

You can add title, description, and tags to timelines:

```python
# Add title/description to timeline metadata
timeline.videoai_metadata = {
    "title_desc": {
        "title": "My Video",
        "description": "This is a description of my video",
        "tags": ["tutorial", "example"],
        "metadata": {
            "version": "1.0",
            "generated": True,
            "custom_field": "Custom value"
        }
    }
}

# Save the timeline with the metadata
timeline_mgr.save_timeline(timeline, "my_timeline")
```

The metadata is preserved during serialization and automatically validated against the schema, which now supports:
- Title (string)
- Description (string with multiple lines)
- Tags (array of strings)
- Additional custom metadata fields

### Path Handling for Title/Description Files

You can also save title/description data separately for use with upload systems:

```python
# Save title/description separately to a standard location
from file_manager import FileManager
file_mgr = FileManager()
title_desc_path = file_mgr.get_title_desc_path(channel_number)
file_mgr.write_json(title_desc_path, timeline.videoai_metadata["title_desc"])
```

The `FileManager.get_title_desc_path()` method provides standardized path resolution:

```python
# Get the standard path for title/description data
title_desc_path = file_mgr.get_title_desc_path(channel_number)  # Default: title_desc.json
custom_path = file_mgr.get_title_desc_path(channel_number, "custom_name")  # custom_name.json
```

### Visualization with Title/Description

Timeline visualizations automatically include title/description metadata when available:

```python
# Create a detailed visualization showing title/description
viz = timeline_mgr.visualize_timeline(timeline, detail_level="detailed")
print(viz)

# Export visualization to a file
timeline_mgr.export_timeline_visualization(
    timeline, 
    output_path=viz_path, 
    detail_level="detailed"
)
```

Example output with title/description:
```
Timeline v3 visualization
Duration: 180 frames, 6.00 seconds @ 30.0 fps
Resolution: 1080x1920, Samplerate: 48000 Hz

Title: My Video
Description: This is a description of my video
Tags: tutorial, example

Sources:
  1. clip1.mp4 (1080x1920, 5.00s)
  2. clip2.mp4 (1080x1920, 6.00s)
...
```

### Accessing Metadata in Timelines

You can easily access metadata from loaded timelines:

```python
# Load a timeline
timeline = timeline_mgr.load_timeline("my_timeline")

# Access title and description metadata
if hasattr(timeline, 'videoai_metadata') and 'title_desc' in timeline.videoai_metadata:
    title_desc = timeline.videoai_metadata['title_desc']
    title = title_desc.get('title', 'Untitled')
    description = title_desc.get('description', '')
    tags = title_desc.get('tags', [])
    
    print(f"Title: {title}")
    print(f"Description: {description}")
    print(f"Tags: {', '.join(tags)}")
```

## Advanced Usage

For more advanced usage, see:
- `examples/timeline_example.py` - Examples of timeline creation and manipulation
- `tests/test_timeline_manager.py` - Test cases showing API usage

## Future Extensions

- Export to professional editor formats (FCP, Shotcut, etc.)
- Interactive timeline editing
- Multi-track composition with graphics and transitions
- Timeline-based rendering alternatives
- Remote timeline storage and sharing
- Timeline versioning and history tracking
- Timeline merge operations for collaborative editing