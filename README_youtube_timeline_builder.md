# YouTube Timeline Builder

This module provides functionality to convert processed YouTube clips into timeline files compatible with the main VideoAI pipeline.

## Features

- Convert YouTube clips to V3 timelines with proper metadata
- Support for captions as text elements
- Channel-specific path handling
- Integration with the existing timeline system

## Usage

### Command Line

```bash
# Download a video, process clips, and generate a timeline
python yt_clips.py --query "NASA Solar System" --generate-timeline --timeline-name "nasa_clips"

# Specify output dimensions and framerate
python yt_clips.py --query "NASA Solar System" --generate-timeline --timeline-width 1280 --timeline-height 720 --timeline-framerate 24

# Specify channel for context
python yt_clips.py --query "NASA Solar System" --generate-timeline --channel 1
```

### API Usage

#### Option 1: Generate timeline directly from processing

```python
from yt_clips import process_video, ProcessingOptions

options = ProcessingOptions(
    generate_timeline=True,
    timeline_name="my_timeline",
    channel_number=1
)
result = process_video("NASA Solar System", options)
json_path, metadata, timeline_path = result
```

#### Option 2: Generate timeline from existing metadata

```python
# Load existing metadata
import json
from pathlib import Path
from yt_clips import VideoMetadata

with open("metadata.json", "r") as f:
    metadata_dict = json.load(f)
    
# Convert to VideoMetadata
metadata = VideoMetadata.parse_obj(metadata_dict)

# Create timeline
from youtube_timeline_builder import YouTubeTimelineBuilder

builder = YouTubeTimelineBuilder(channel_number=1)
timeline = builder.create_timeline_from_clips(metadata)
timeline_path = builder.save_timeline(timeline, timeline_name="my_timeline")
```

## Class Reference

### YouTubeTimelineBuilder

The main class for converting YouTube clips to timelines.

#### Constructor

```python
def __init__(self, timeline_manager=None, channel_number=None)
```

- `timeline_manager`: Optional TimelineManager instance (creates one if None)
- `channel_number`: Optional channel number for context-specific operations

#### Methods

##### create_timeline_from_clips

```python
def create_timeline_from_clips(self, metadata, width=1920, height=1080, framerate=30.0)
```

Converts processed YouTube clips to a V3 timeline.

- `metadata`: VideoMetadata object containing clip information
- `width`: Output video width
- `height`: Output video height
- `framerate`: Output framerate

Returns a V3 timeline object.

##### save_timeline

```python
def save_timeline(self, timeline, output_path=None, timeline_name=None)
```

Saves a timeline to disk.

- `timeline`: Timeline object to save
- `output_path`: Optional explicit path to save to
- `timeline_name`: Optional name to use when generating path

Returns the path where the timeline was saved.

## Integration with yt_clips.py

The YouTubeTimelineBuilder is integrated with yt_clips.py, allowing for direct timeline generation during YouTube clip processing. The following options have been added to ProcessingOptions:

- `generate_timeline`: Whether to generate a timeline (default: False)
- `timeline_name`: Name for the generated timeline (default: None, generates a name based on timestamp)
- `timeline_width`: Width of the output timeline (default: 1920)
- `timeline_height`: Height of the output timeline (default: 1080)
- `timeline_framerate`: Framerate of the output timeline (default: 30.0)
- `channel_number`: Channel number for context-specific operations (default: None)

## Implementation Details

### Caption Integration

Captions from YouTube clips are automatically integrated into the timeline as text elements with proper timing and styling. Text elements are placed on a separate video track (track 1) for flexibility in the editing pipeline.

### Metadata Handling

The generated timelines include detailed metadata about the source YouTube video and the processing parameters, facilitating proper integration with the main VideoAI pipeline:

```json
"videoai_metadata": {
    "version": "1.0",
    "type": "v3",
    "created_at": "2023-01-01T12:00:00",
    "description": "YouTube clips timeline for Example Video",
    "channel": 1,
    "youtube_info": {
        "video_id": "abc123",
        "title": "Example Video",
        "url": "https://www.youtube.com/watch?v=abc123",
        "clip_count": 5,
        "processor_version": "1.0.0"
    }
}
```

## Error Handling

The YouTubeTimelineBuilder implements comprehensive error handling with custom exceptions and detailed logging to facilitate debugging and ensure graceful degradation.

## Dependencies

- timeline_manager.py: For creating and serializing timelines
- file_manager.py: For file operations and path handling
- auto_editor: For timeline structures and video handling
- yt_clips.py: For YouTube video models and clip processing