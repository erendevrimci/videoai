# Timeline System in VideoAI

## Overview

The timeline system is a sophisticated framework designed to create, manage, and visualize structured representations of video editing decisions in the VideoAI project. It serves as an abstraction layer between the script generation and the final video production, providing a more flexible and powerful way to organize video content.

## Core Concepts

### What is a Timeline?

A timeline in VideoAI is a structured representation of a video project that includes:

- **Tracks**: Separate layers for video and audio content
- **Media Elements**: Video clips, audio segments, images, and other visual elements
- **Timing Information**: Precise frame-level positioning of each element
- **Metadata**: Production details, script information, and other context

The timeline system transforms the linear script-to-video pipeline into a more flexible, non-destructive editing process where decisions can be reviewed and modified at any stage.

### Timeline Types

Two main timeline formats are supported:

1. **v1 Timeline**: A simple timeline based on chunks of a single source video
   - Primarily used for basic cuts of a single video
   - Stores segments as start/end frame pairs
   - Example structure:
     ```
     {
       "source": "/path/to/video.mp4",
       "chunks": [[0, 90], [150, 230], [240, 300]],
       "videoai_metadata": { ... }
     }
     ```

2. **v3 Timeline**: An advanced timeline supporting multiple tracks, sources, and media types
   - Supports multiple video and audio tracks
   - Can include different media types (video, audio, images, shapes)
   - Stores detailed metadata about the production process
   - Example structure:
     ```
     {
       "timebase": "30/1",
       "samplerate": 48000,
       "resolution": [1920, 1080],
       "background": "#000000",
       "v": [
         [{"start": 0, "dur": 90, "src": "/path/to/clip1.mp4", ...}]
       ],
       "a": [
         [{"start": 0, "dur": 90, "src": "/path/to/audio.mp3", ...}]
       ],
       "videoai_metadata": { ... }
     }
     ```

## Benefits of the Timeline System

The timeline system offers several advantages over the traditional linear approach:

1. **Non-destructive Editing**: Changes to the timeline don't affect the original media files
2. **Flexibility**: Easily modify, rearrange, or replace elements without starting from scratch
3. **Progressive Building**: Build the timeline incrementally through each production step
4. **Structured Representation**: Maintain relationships between script, voice, and visuals
5. **Advanced Editing**: Support for multi-track compositions, transitions, and effects
6. **Visualization**: Better understanding of the production process through timeline visualizations
7. **Integration Potential**: Future compatibility with professional editing tools

## Timeline Creation and Management

### The TimelineManager Class

The central component of the timeline system is the `TimelineManager` class, which provides:

- Timeline creation methods (`create_v1_timeline`, `create_v3_timeline`)
- Conversion methods between formats (`clip_sequence_to_timeline`, `convert_v1_to_v3`)
- Serialization and deserialization to/from JSON
- Visualization capabilities (`visualize_timeline`, `export_timeline_visualization`)
- Channel-aware file path handling

```python
from timeline_manager import TimelineManager

# Initialize with channel-specific settings
timeline_mgr = TimelineManager(channel_number=1)

# Create a new timeline
timeline = timeline_mgr.create_v3_timeline(
    width=1080,
    height=1920,
    framerate=30
)

# Save the timeline to a file
timeline_mgr.save_timeline(timeline, "my_timeline")
```

### Timeline Elements

A timeline consists of:

- **Video tracks** (`timeline.v`): Lists of video, image, and rectangle objects
- **Audio tracks** (`timeline.a`): Lists of audio objects
- **Metadata** (`timeline.videoai_metadata`): Production information and channel details
- **Timeline properties**: timebase, samplerate, resolution, background

Each media element has specific properties:

- **TlVideo**: Video clips with start, duration, source file, offset, speed
- **TlAudio**: Audio clips with similar properties plus volume
- **TlImage**: Static images with position and opacity
- **TlRect**: Colored rectangles for visual elements

## Integration With Video Production Flow

The timeline system integrates with the entire video production pipeline:

### 1. Script Generation

- Timeline created at the beginning of the process
- Script segments are added to the timeline as metadata
- Timeline saved as "script_timeline.json"

### 2. Voice-Over Generation

- Voice-over audio is added to the timeline audio tracks
- Timeline updated with voice duration and metadata
- Timeline saved as "voice_timeline.json"

### 3. Caption Generation

- SRT files are parsed to extract precise word timings
- Caption segments are added to the timeline metadata
- Timeline saved as "captions_timeline.json"

### 4. Video Editing

- Video clips are added to the timeline's video tracks
- Each clip becomes a TlVideo object with source, timing, and position
- Timeline can be rendered directly or using traditional methods
- Timeline saved as "final_timeline.json"*

### 5. Title and Description Generation

- Title and description added to timeline metadata
- Facilitates YouTube upload integration
- Timeline saved as "metadata_timeline.json"*

### 6. YouTube Upload

- Upload results and video ID added to timeline metadata
- Timeline saved as "published_timeline.json"*

## Using the Timeline System

### Enabling Timeline Support

To use the timeline-aware pipeline, add the `--timeline` flag when running the main script:

```bash
python main.py --channel 1 --timeline
```

This flag enables the `process_channel_with_timeline` function, which creates a timeline at the beginning of the process and builds it progressively through each step.

### Converting Clip Sequences to Timelines

The existing clip-based approach can be converted to timelines:

```python
# Define a clip sequence (from video_edit.py)
clip_sequence = [
    {
        "clip_name": "sample_clips/clip1.mp4",
        "start_time": 0,
        "duration": 5.0,
        "script_segment": "First segment of the script"
    },
    # More clips...
]

# Convert to timeline
timeline = timeline_mgr.clip_sequence_to_timeline(
    clip_sequence,
    output_width=1080,
    output_height=1920
)
```

### Timeline Visualization

One of the powerful features of the timeline system is the ability to visualize timelines:

```python
# Generate a text visualization
visualization = timeline_mgr.visualize_timeline(timeline, detail_level="detailed")
print(visualization)

# Export visualization to a file
timeline_mgr.export_timeline_visualization(timeline, output_path="timeline_viz.txt")
```

Example visualization:
```
Timeline v3 visualization
Duration: 210 frames, 7.00 seconds @ 30.0 fps
Resolution: 1920x1080, Samplerate: 48000 Hz

Video Tracks:
Track 1:
+===================++===============================================+
1                    2                                                

  1. Video: 0.00s-3.00s (3.00s) | Source: clip1.mp4
  2. Video: 3.00s-7.00s (4.00s) | Source: clip2.mp4

Audio Tracks:
Track 1:
+~~~~~~~~~~~~~~~~~~~~~~~~~~~++~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
1                            2                                

  1. Audio: 0.00s-3.00s (3.00s) | Source: voice1.mp3
  2. Audio: 3.00s-7.00s (4.00s) | Source: voice2.mp3
```

## Configuration

The timeline system is highly configurable through the `TimelineConfig` class in `config.py`:

```python
class TimelineConfig(BaseModel):
    # Timeline storage locations
    storage_directory: str = Field(default="timelines")
    visualization_directory: str = Field(default="timeline_visualizations")
    
    # Default timeline settings
    default_framerate: int = Field(default=30)
    default_width: int = Field(default=1080)
    default_height: int = Field(default=1920)
    default_samplerate: int = Field(default=48000)
    default_background: str = Field(default="#000000")
    
    # Visualization settings
    visualization: TimelineVisualizationConfig = Field(default_factory=TimelineVisualizationConfig)
    
    # Serialization settings
    serialization: TimelineSerializationConfig = Field(default_factory=TimelineSerializationConfig)
```

Channel-specific overrides can be defined in the `ChannelTimelineConfig` class and applied to specific channels.

## Differences from Previous Approach

The timeline system differs from the previous approach in several ways:

### Previous Approach

1. **Linear Process**: Script → Voice → Clips → Video in a fixed sequence
2. **Direct File Dependencies**: Each step depends on specific file paths from previous steps
3. **Limited Flexibility**: Changes to one step often require redoing all subsequent steps
4. **Simple Clip Format**: Basic start/end information without rich metadata
5. **Single Output**: Focused only on creating a single video file

### Timeline Approach

1. **Non-linear Process**: Timeline created early and built progressively
2. **Structured Representation**: All information stored in a unified timeline object
3. **Flexibility**: Changes can be made to any aspect without complete rework
4. **Rich Metadata**: Preserves relationships between script, voice, and visuals
5. **Multiple Outputs**: Can generate various outputs (video, visualizations, exports)

## Future Enhancements

The timeline system is designed to be extensible for future capabilities:

1. **Advanced Editing Features**
   - Transitions between clips
   - Multi-layer compositing
   - Motion graphics and animations
   - Color grading and visual effects
   - Applying LUT's to the clips

2. **Export to Professional Formats**
   - Final Cut Pro XML
   - Adobe Premiere Pro project files
   - DaVinci Resolve project files

3. **Interactive Timeline Editing**
   - GUI-based timeline manipulation
   - Drag-and-drop clip arrangement
   - Real-time preview capabilities

4. **Remote Collaboration**
   - Timeline versioning and history
   - Cloud-based timeline storage
   - Collaborative editing workflows

## Getting Started with Timelines

To start using the timeline system:

1. Enable timeline support with the `--timeline` flag:
   ```bash
   python main.py --channel 1 --timeline
   ```

2. Check the generated timeline files in the `timelines/` directory

3. Visualize timelines to better understand the process:
   ```bash
   python timeline_visualize.py --channel 1 --timeline script_timeline
   ```

4. Explore the example files in the `examples/` directory:
   - `timeline_example.py`: Basic timeline creation and manipulation
   - `timeline_config_example.py`: Custom timeline configuration

## Conclusion

The timeline system represents a significant evolution in the VideoAI project's architecture, providing a more flexible, powerful, and structured approach to video production. By abstracting the editing process into a timeline representation, it enables more advanced features while maintaining compatibility with the existing pipeline.