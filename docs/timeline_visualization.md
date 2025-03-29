# Timeline Visualization

This document explains the timeline visualization capabilities of the VideoAI project's TimelineManager. The timeline visualization system provides ASCII-based representations of timelines for easy review and debugging.

## Visualization Features

- **Multiple Detail Levels**: Choose from minimal, normal, or detailed visualizations
- **Proportional Representation**: Clips are shown with width proportional to duration
- **Time Markers**: Clear time indicators for navigating the timeline
- **Multi-track Support**: Visualize complex v3 timelines with multiple video and audio tracks
- **Export Capabilities**: Save visualizations to text files for documentation and sharing
- **Console Output**: Print summaries and visualizations directly to the console

## Detail Levels

The timeline visualization system supports three detail levels:

### 1. Minimal

A compact representation showing only the essential information:
- Timeline source and duration
- Simple visualization of the clips
- Minimal metadata

Example:
```
Timeline: sample_video.mp4 (300 frames)

Chunks:
Mode: Keep segments

--------------------------------------------------------------------------------
 1 | ##############                                                             
 2 |                         ###########                                        
 3 |                                             ############                   
--------------------------------------------------------------------------------
```

### 2. Normal

A balanced visualization with more details:
- Timeline source, duration, and basic metadata
- Visualization of clips with time markers
- Clip start and end frames
- Basic clip information

Example:
```
Timeline visualization for: sample_video.mp4
Duration: 300 frames, 10.00 seconds @ 30.00 fps

Chunks:
Mode: Keep segments

--------------------------------------------------------------------------------
|                   |                   |                   |                   |
0s                2.5s                5.0s                7.5s               10.0s

 1 | ############### |                 | 0-90
 2 |                 | ############# | | 150-230
 3 |                 |                 | ###########       | 240-300
--------------------------------------------------------------------------------
```

### 3. Detailed

A comprehensive visualization with full details:
- Complete source metadata (resolution, codec, audio)
- Detailed time markers
- Frame and time information for each clip
- Additional statistics and summaries

Example:
```
Timeline visualization for: sample_video.mp4
Duration: 300 frames, 10.00 seconds @ 30.00 fps

Source details:
  - Resolution: 1920x1080
  - Codec: h264
  - Audio: 48000 Hz, 2 channels

Chunks:
Mode: Keep segments

--------------------------------------------------------------------------------
|                   |                   |                   |                   |
0s                2.5s                5.0s                7.5s               10.0s

 1 | ############### |                 | Frames: 0-90 | Time: 0.00s-3.00s | Duration: 3.00s
   |
 2 |                 | ############# | | Frames: 150-230 | Time: 5.00s-7.67s | Duration: 2.67s
   |
 3 |                 |                 | ########### | Frames: 240-300 | Time: 8.00s-10.00s | Duration: 2.00s
--------------------------------------------------------------------------------

Summary:
  - Total timeline frames: 300
  - Kept frames: 230 (76.7%)
  - Cut frames: 70 (23.3%)
```

## V3 Timeline Visualization

For v3 timelines (multi-track), the visualization shows:
- Timeline metadata (duration, resolution, sample rate)
- Source information
- Multiple video tracks with clip representation
- Audio tracks (in normal and detailed views)
- Clip details based on the selected detail level

Example:
```
Timeline v3 visualization
Duration: 210 frames, 7.00 seconds @ 30.0 fps
Resolution: 1920x1080, Samplerate: 48000 Hz

--------------------------------------------------------------------------------
|                   |                   |                   |                   |
0s                1.8s                3.5s                5.2s                7.0s

Video Tracks:
Track 1:
+==========================++===============================================+
1                           2                                                

  1. Video: 0.00s-3.00s (3.00s) | Source: sample_video.mp4
  2. Video: 3.00s-7.00s (4.00s) | Source: sample_video.mp4

Track 2:
                  +RRRRRRRRRR+                                              
                  1                                                         

  1. Rectangle: 1.00s-3.00s (2.00s) | Color: #FF0000

Audio Tracks:
Track 1:
+~~~~~~~~~~~~~~~~~~~~~~~~~~~+                                              
1                                                                          

  1. Audio: 0.00s-3.00s (3.00s) | Source: sample_video.mp4
```

## Usage Examples

### Basic Visualization

```python
from timeline_manager import TimelineManager

# Initialize timeline manager
timeline_mgr = TimelineManager(channel_number=1)

# Load a timeline
timeline = timeline_mgr.load_timeline("my_timeline")

# Visualize with default settings (normal detail level)
visualization = timeline_mgr.visualize_timeline(timeline)
print(visualization)

# Print a compact summary
timeline_mgr.print_timeline_summary(timeline)
```

### Custom Visualization

```python
# Visualize with detailed information and wider display
detailed_viz = timeline_mgr.visualize_timeline(
    timeline,
    width=120,  # Wider display
    detail_level='detailed'  # Full details
)
print(detailed_viz)
```

### Exporting Visualizations

```python
# Export visualization to a text file
timeline_mgr.export_timeline_visualization(
    timeline,
    output_path="/path/to/output/timeline_viz.txt",
    width=100,
    detail_level='detailed'
)

# Export using default path (in channel output directory)
timeline_mgr.export_timeline_visualization(
    timeline,
    detail_level='normal'
)
```

## Integration with Video Editing

Timeline visualizations can be helpful during the video editing process:
- Review clip arrangements before rendering
- Debug complex timelines
- Document editing decisions
- Share editing plans with team members

When working with the VideoAI pipeline, visualizations can be generated at any point to understand the current state of the timeline before proceeding to the next step.