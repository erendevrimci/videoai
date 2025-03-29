# Phase 1: YouTubeTimelineBuilder Implementation Action Plan

## Overview

This document details the implementation plan for Phase 1 of the asynchronous integration of `yt_clips.py` with the main VideoAI pipeline. Phase 1 focuses on enhancing `yt_clips.py` with timeline generation capabilities, allowing it to produce timeline files compatible with the main pipeline's export formats.

## Goals

1. Create the `YouTubeTimelineBuilder` class for transforming YouTube clips into V3 timelines
2. Add caption integration with proper timing and styling
3. Implement channel-specific path handling
4. Update `yt_clips.py` to support timeline generation mode
5. Ensure compatibility with the existing TimelineManager and FileManager systems

## Tasks and Implementation Details

### 1. Create YouTubeTimelineBuilder Class

#### 1.1 Class Structure and Dependencies

- [ ] Create a new file `youtube_timeline_builder.py` in the project root
- [ ] Import required components:
  - TimelineManager for timeline manipulation
  - FileManager for path handling
  - Auto-editor components (v3, TlVideo, TlText, etc.)
  - Logging components

```python
# youtube_timeline_builder.py
"""
YouTubeTimelineBuilder module for VideoAI project.

Creates timeline files from processed YouTube clips with proper
formatting for integration with the main pipeline.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from fractions import Fraction

# Import auto-editor components
from auto_editor.timeline import v3, TlVideo
from auto_editor.ffwrapper import initFileInfo

# Import VideoAI components
from file_manager import FileManager
from timeline_manager import TimelineManager, TlText
from logging_system.logger import Logger
from logging_system.exception_handler import log_exceptions, VideoAIException

# Import yt_clips models
from yt_clips import VideoMetadata, VideoClip, CaptionSegment

# Initialize logger
logger = Logger.get_logger("youtube_timeline_builder")

# Initialize managers
file_mgr = FileManager()
```

#### 1.2 Implement Core Builder Class

- [ ] Create `YouTubeTimelineBuilder` class with initialization and core methods
- [ ] Implement `create_timeline_from_clips` method
- [ ] Add helper methods for clip conversion and metadata handling
- [ ] Ensure proper error handling with custom exceptions

```python
class YouTubeTimelineError(VideoAIException):
    """Base exception for YouTube timeline building errors."""
    pass

class YouTubeTimelineBuilder:
    """
    Creates timeline files from YouTube clips.
    """
    def __init__(self, 
                 timeline_manager: Optional[TimelineManager] = None,
                 channel_number: Optional[int] = None):
        """
        Initialize with optional timeline manager and channel context.
        
        Args:
            timeline_manager: Timeline manager instance (creates one if None)
            channel_number: Channel number for context-specific operations
        """
        self.timeline_manager = timeline_manager or TimelineManager(channel_number=channel_number)
        self.channel_number = channel_number
        
    @log_exceptions(logger_instance=logger)
    def create_timeline_from_clips(self, 
                                  metadata: VideoMetadata, 
                                  width: int = 1920, 
                                  height: int = 1080, 
                                  framerate: float = 30.0) -> v3:
        """
        Convert processed YouTube clips to a v3 timeline.
        
        Args:
            metadata: VideoMetadata object containing clip information
            width: Output video width
            height: Output video height
            framerate: Output framerate
            
        Returns:
            v3 timeline object
            
        Raises:
            YouTubeTimelineError: If timeline creation fails
        """
        logger.info(f"Creating timeline from {len(metadata.clips)} YouTube clips")
        
        try:
            # Create empty timeline with appropriate dimensions
            timeline = self.timeline_manager.create_v3_timeline(
                width=width,
                height=height,
                framerate=framerate
            )
            
            # Add clips to timeline with proper timing
            current_frame = 0
            
            for clip_idx, clip in enumerate(metadata.clips):
                # Add video clip to timeline
                current_frame = self._add_clip_to_timeline(
                    timeline=timeline,
                    clip=clip,
                    clip_idx=clip_idx,
                    current_frame=current_frame,
                    framerate=framerate
                )
            
            # Add metadata about the source video
            self._add_metadata_to_timeline(timeline, metadata)
            
            logger.info(f"Successfully created timeline with {len(metadata.clips)} clips")
            return timeline
            
        except Exception as e:
            logger.error(f"Error creating timeline: {e}", exc_info=True)
            raise YouTubeTimelineError(f"Failed to create timeline from clips", cause=e)
```

#### 1.3 Implement Helper Methods

- [ ] Add private helper methods for clip and caption processing
- [ ] Implement metadata handling with proper versioning
- [ ] Ensure all paths are properly normalized

```python
def _add_clip_to_timeline(self, 
                          timeline: v3, 
                          clip: VideoClip, 
                          clip_idx: int,
                          current_frame: int, 
                          framerate: float) -> int:
    """
    Add a clip to the timeline and return the updated frame position.
    
    Args:
        timeline: Target timeline
        clip: Clip to add
        clip_idx: Index of the clip
        current_frame: Current position in the timeline (frames)
        framerate: Timeline framerate
        
    Returns:
        Updated frame position after adding this clip
    """
    clip_path = clip.path
    duration_frames = int(clip.duration * framerate)
    
    # Normalize the path
    norm_path = file_mgr.normalize_path(clip_path)
    
    # Create TlVideo object for this clip
    video_obj = TlVideo(
        start=current_frame,
        dur=duration_frames,
        src=initFileInfo(str(norm_path)),
        offset=0,  # Start from beginning of clip file
        speed=1.0,
        stream=0
    )
    
    # Add to the first video track
    timeline.v[0].append(video_obj)
    
    # Add caption segments as text elements on the second track
    if len(timeline.v) < 2:
        timeline.v.append([])  # Add second track for captions
        
    # Process captions if available
    for caption in clip.captions:
        self._add_caption_to_timeline(
            timeline=timeline,
            caption=caption,
            clip=clip,
            current_frame=current_frame,
            framerate=framerate,
            width=timeline.res[0],
            height=timeline.res[1]
        )
    
    # Return updated position
    return current_frame + duration_frames

def _add_caption_to_timeline(self,
                             timeline: v3,
                             caption: CaptionSegment,
                             clip: VideoClip,
                             current_frame: int,
                             framerate: float,
                             width: int,
                             height: int):
    """
    Add a caption segment to the timeline.
    
    Args:
        timeline: Target timeline
        caption: Caption segment to add
        clip: Parent clip
        current_frame: Current position in timeline (frames)
        framerate: Timeline framerate
        width: Timeline width
        height: Timeline height
    """
    # Calculate timing in the timeline
    # Adjust caption timing relative to clip start time
    caption_start = current_frame + int((caption.start - clip.start_time) * framerate)
    caption_end = current_frame + int((caption.end - clip.start_time) * framerate)
    caption_duration = caption_end - caption_start
    
    # Skip invalid captions
    if caption_duration <= 0:
        logger.warning(f"Skipping caption with invalid duration: {caption_duration} frames")
        return
    
    # Create text element for caption
    text_obj = TlText(
        start=caption_start,
        dur=caption_duration,
        text=caption.text,
        x=width // 2,  # Center horizontally
        y=height - 100,  # Position near bottom
        font="Arial",
        font_size=36,
        color="#FFFFFF",
        bg_color="#00000080",  # Semi-transparent black background
        align="center"
    )
    
    # Add to second video track
    timeline.v[1].append(text_obj)

def _add_metadata_to_timeline(self, timeline: v3, metadata: VideoMetadata):
    """
    Add YouTube-specific metadata to the timeline.
    
    Args:
        timeline: Target timeline
        metadata: YouTube video metadata
    """
    timeline.videoai_metadata = {
        'version': '1.0',
        'type': 'v3',
        'created_at': datetime.now().isoformat(),
        'description': f'YouTube clips timeline for {metadata.video.title}',
        'channel': self.channel_number,
        'youtube_info': {
            'video_id': metadata.video.video_info.video_id,
            'title': metadata.video.title,
            'url': str(metadata.video.url),
            'clip_count': len(metadata.clips),
            'processor_version': metadata.processor_version
        }
    }
```

#### 1.4 Add Serialization Methods

- [ ] Add methods to save timelines to JSON files
- [ ] Implement functionality to load and modify existing timelines
- [ ] Add validation to ensure timeline integrity

```python
@log_exceptions(logger_instance=logger)
def save_timeline(self, 
                 timeline: v3, 
                 output_path: Optional[Union[str, Path]] = None,
                 timeline_name: Optional[str] = None) -> Path:
    """
    Save a timeline to disk.
    
    Args:
        timeline: Timeline object to save
        output_path: Optional explicit path to save to
        timeline_name: Optional name to use when generating path
        
    Returns:
        Path where the timeline was saved
        
    Raises:
        YouTubeTimelineError: If saving fails
    """
    try:
        # Determine the output path
        if output_path is not None:
            # Use explicitly provided path
            save_path = file_mgr.normalize_path(output_path)
        elif timeline_name is not None:
            # Generate path based on name and channel
            save_path = self.timeline_manager.get_timeline_path(timeline_name)
        else:
            # Use default name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.channel_number is not None:
                default_name = f"youtube_clips_channel_{self.channel_number}_{timestamp}"
            else:
                default_name = f"youtube_clips_{timestamp}"
            save_path = self.timeline_manager.get_timeline_path(default_name)
        
        # Save the timeline
        logger.info(f"Saving timeline to {save_path}")
        
        # Use timeline manager to serialize
        self.timeline_manager.serialize_timeline(
            timeline=timeline, 
            path=save_path,
            description="YouTube clips timeline"
        )
        
        logger.info(f"Timeline saved successfully to {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Error saving timeline: {e}", exc_info=True)
        raise YouTubeTimelineError(f"Failed to save timeline", cause=e)
```

### 2. Update yt_clips.py with Timeline Mode

#### 2.1 Add Timeline Generation Option

- [ ] Add timeline mode flag to ProcessingOptions in yt_clips.py
- [ ] Implement timeline output path option
- [ ] Add configuration for timeline parameters (resolution, framerate)

```python
# Update ProcessingOptions in yt_clips.py to add:
class ProcessingOptions(BaseModel):
    """Options for video processing."""
    # Existing options...
    
    # New timeline-related options
    generate_timeline: bool = False
    timeline_name: Optional[str] = None
    timeline_width: int = 1920
    timeline_height: int = 1080
    timeline_framerate: float = 30.0
    channel_number: Optional[int] = None
```

#### 2.2 Integrate Timeline Generation into Main Process

- [ ] Update the `process_video` function to generate timelines when requested
- [ ] Add proper error handling and logging for timeline generation
- [ ] Ensure proper path handling with channel context

```python
# Add to process_video function in yt_clips.py:

# After processing clips and generating captions
if options.generate_timeline:
    try:
        logger.info("Generating timeline from clips")
        
        # Import the builder (delayed import to avoid circular dependencies)
        from youtube_timeline_builder import YouTubeTimelineBuilder
        
        # Create builder with channel context
        builder = YouTubeTimelineBuilder(channel_number=options.channel_number)
        
        # Create timeline
        timeline = builder.create_timeline_from_clips(
            metadata=metadata,
            width=options.timeline_width,
            height=options.timeline_height,
            framerate=options.timeline_framerate
        )
        
        # Save timeline
        timeline_path = builder.save_timeline(
            timeline=timeline,
            timeline_name=options.timeline_name
        )
        
        logger.info(f"Timeline generated and saved to: {timeline_path}")
        
        # Add timeline path to return data
        return json_path, metadata, timeline_path
    except Exception as e:
        logger.error(f"Error generating timeline: {e}", exc_info=True)
        # Continue with normal return even if timeline generation fails
        return json_path, metadata
else:
    # Original return
    return json_path, metadata
```

#### 2.3 Update Command Line Interface

- [ ] Add command line arguments for timeline generation
- [ ] Update help text and documentation
- [ ] Ensure backward compatibility

```python
# Update in main() function of yt_clips.py:

# Add new arguments to argparse
parser.add_argument("--generate-timeline", action="store_true", 
                    help="Generate a timeline from processed clips")
parser.add_argument("--timeline-name", type=str, 
                    help="Name for the generated timeline")
parser.add_argument("--timeline-width", type=int, default=1920,
                    help="Width of the output timeline")
parser.add_argument("--timeline-height", type=int, default=1080,
                    help="Height of the output timeline")
parser.add_argument("--timeline-framerate", type=float, default=30.0,
                    help="Framerate of the output timeline")
parser.add_argument("--channel", type=int, 
                    help="Channel number for context-specific operations")

# Update ProcessingOptions
options = ProcessingOptions(
    # Existing options...
    generate_timeline=args.generate_timeline,
    timeline_name=args.timeline_name,
    timeline_width=args.timeline_width,
    timeline_height=args.timeline_height,
    timeline_framerate=args.timeline_framerate,
    channel_number=args.channel
)

# Handle different return values based on timeline generation
result = process_video(search_query, options)
if result:
    if options.generate_timeline and len(result) > 2:
        json_path, metadata, timeline_path = result
        logger.info(f"Processing complete. Metadata saved to: {json_path}")
        logger.info(f"Timeline saved to: {timeline_path}")
        print(f"Processing complete. Metadata saved to: {json_path}")
        print(f"Timeline saved to: {timeline_path}")
    else:
        json_path, metadata = result
        logger.info(f"Processing complete. Metadata saved to: {json_path}")
        print(f"Processing complete. Metadata saved to: {json_path}")
    print(f"Downloaded: {metadata.video.title}")
    print(f"Processed {len(metadata.clips)} clips with a total of {metadata.total_caption_segments()} caption segments")
```

### 3. Implement Unit Tests

#### 3.1 Create Test Cases for YouTubeTimelineBuilder

- [ ] Create `tests/test_youtube_timeline_builder.py` file
- [ ] Implement tests for timeline creation
- [ ] Test caption integration
- [ ] Test metadata handling

```python
# tests/test_youtube_timeline_builder.py
import unittest
from pathlib import Path
import json
from datetime import datetime

# Mock imports and setup
from unittest.mock import MagicMock, patch
from youtube_timeline_builder import YouTubeTimelineBuilder

class TestYouTubeTimelineBuilder(unittest.TestCase):
    """Test cases for YouTubeTimelineBuilder class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock objects
        self.mock_timeline_manager = MagicMock()
        self.mock_file_manager = MagicMock()
        
        # Create a builder instance with mocks
        self.builder = YouTubeTimelineBuilder(self.mock_timeline_manager)
        
        # Create mock metadata
        self.mock_metadata = self._create_mock_metadata()
        
    def _create_mock_metadata(self):
        """Create mock VideoMetadata for testing."""
        # Create mock objects with the same structure as the real ones
        # This avoids dependency on the actual yt_clips models
        mock_metadata = MagicMock()
        
        # Configure the mock
        mock_metadata.video.title = "Test YouTube Video"
        mock_metadata.video.url = "https://www.youtube.com/watch?v=test123"
        mock_metadata.video.video_info.video_id = "test123"
        mock_metadata.processor_version = "1.0.0"
        
        # Create mock clips
        mock_metadata.clips = []
        for i in range(3):
            mock_clip = MagicMock()
            mock_clip.path = Path(f"/tmp/test_clip_{i}.mp4")
            mock_clip.start_time = i * 10.0
            mock_clip.end_time = (i + 1) * 10.0
            mock_clip.duration = 10.0
            
            # Add mock captions
            mock_clip.captions = []
            for j in range(2):
                mock_caption = MagicMock()
                mock_caption.start = i * 10.0 + j * 4.0
                mock_caption.end = i * 10.0 + (j + 1) * 4.0
                mock_caption.text = f"Test caption {i}-{j}"
                mock_clip.captions.append(mock_caption)
                
            mock_metadata.clips.append(mock_clip)
            
        return mock_metadata
    
    @patch('youtube_timeline_builder.initFileInfo')
    def test_create_timeline_from_clips(self, mock_init_file_info):
        """Test creating a timeline from clips."""
        # Configure mocks
        mock_timeline = MagicMock()
        mock_timeline.v = [[]]  # Empty video track
        mock_timeline.res = [1920, 1080]
        
        self.mock_timeline_manager.create_v3_timeline.return_value = mock_timeline
        
        # Add a second video track for captions
        mock_timeline.v.append([])
        
        # Configure file info mock
        mock_init_file_info.return_value = MagicMock()
        
        # Call the method under test
        result = self.builder.create_timeline_from_clips(
            metadata=self.mock_metadata,
            width=1920,
            height=1080,
            framerate=30.0
        )
        
        # Verify the result
        self.assertEqual(result, mock_timeline)
        
        # Verify timeline creation was called
        self.mock_timeline_manager.create_v3_timeline.assert_called_once()
        
        # Verify video objects were added (one per clip)
        self.assertEqual(len(mock_timeline.v[0]), 3)
        
        # Verify caption objects were added (two per clip)
        self.assertEqual(len(mock_timeline.v[1]), 6)
        
        # Verify metadata was added
        self.assertIn('videoai_metadata', mock_timeline.__dict__)
        self.assertIn('youtube_info', mock_timeline.videoai_metadata)
```

#### 3.2 Create Integration Tests

- [ ] Create tests for the full workflow
- [ ] Test command line usage
- [ ] Test error handling and edge cases

```python
# tests/test_youtube_timeline_integration.py
import unittest
from pathlib import Path
import tempfile
import json
import os
import shutil

# Import the modules to test
from yt_clips import process_video, ProcessingOptions
from youtube_timeline_builder import YouTubeTimelineBuilder
from timeline_manager import TimelineManager

class TestYouTubeTimelineIntegration(unittest.TestCase):
    """Integration tests for YouTube timeline generation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.download_dir = Path(self.temp_dir) / "downloads"
        self.output_dir.mkdir()
        self.download_dir.mkdir()
        
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skip("Integration test requiring network access - run manually")
    def test_end_to_end_timeline_generation(self):
        """Test the entire timeline generation process with a real video."""
        # Create processing options
        options = ProcessingOptions(
            max_results=1,
            output_directory=self.output_dir,
            download_directory=self.download_dir,
            scene_threshold=30.0,
            whisper_model="tiny",  # Use tiny model for faster testing
            min_clip_duration=1.0,
            skip_captions=False,
            generate_timeline=True,
            timeline_name="test_timeline",
            timeline_width=1280,
            timeline_height=720,
            timeline_framerate=30.0
        )
        
        # Process a short video (use a known short video for testing)
        result = process_video("NASA Solar System short", options)
        
        # Verify we got the expected return value structure
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # metadata_path, metadata, timeline_path
        
        json_path, metadata, timeline_path = result
        
        # Verify the timeline file exists
        self.assertTrue(timeline_path.exists())
        
        # Load the timeline and verify structure
        timeline_mgr = TimelineManager()
        timeline = timeline_mgr.deserialize_timeline(timeline_path)
        
        # Verify basic timeline properties
        self.assertIsNotNone(timeline)
        self.assertEqual(timeline.res[0], 1280)
        self.assertEqual(timeline.res[1], 720)
        
        # Verify tracks exist with content
        self.assertTrue(len(timeline.v) >= 2)  # At least video and caption tracks
        self.assertTrue(len(timeline.v[0]) > 0)  # Video track has clips
        
        # Verify metadata
        self.assertIn('videoai_metadata', timeline.__dict__)
        self.assertIn('youtube_info', timeline.videoai_metadata)
        self.assertEqual(timeline.videoai_metadata['youtube_info']['video_id'], metadata.video.video_info.video_id)
```

### 4. Documentation and Usability

#### 4.1 Add README.md for YouTubeTimelineBuilder

- [ ] Document the class, methods, and parameters
- [ ] Provide usage examples
- [ ] Document command line options

```markdown
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

```python
from yt_clips import process_video, ProcessingOptions
from youtube_timeline_builder import YouTubeTimelineBuilder

# Option 1: Generate timeline directly from processing
options = ProcessingOptions(
    generate_timeline=True,
    timeline_name="my_timeline",
    channel_number=1
)
result = process_video("NASA Solar System", options)
json_path, metadata, timeline_path = result

# Option 2: Generate timeline from existing metadata
from youtube_timeline_builder import YouTubeTimelineBuilder

# Load existing metadata
import json
from pathlib import Path
with open("metadata.json", "r") as f:
    metadata_dict = json.load(f)
    
# Convert to VideoMetadata (assuming you have the model)
from yt_clips import VideoMetadata
metadata = VideoMetadata.parse_obj(metadata_dict)

# Create timeline
builder = YouTubeTimelineBuilder(channel_number=1)
timeline = builder.create_timeline_from_clips(metadata)
timeline_path = builder.save_timeline(timeline, timeline_name="my_timeline")
```
```

#### 4.2 Update Main Project Documentation

- [ ] Update project README.md with timeline generation information
- [ ] Document the integration with the main pipeline
- [ ] Add examples for common use cases

## Risk Management and Mitigation

### Potential Technical Challenges

1. **Circular Import Dependencies**
   - **Risk**: Circular imports between yt_clips.py, youtube_timeline_builder.py, and timeline_manager.py
   - **Mitigation**: Use delayed imports, import types, or restructure modules to avoid cycles

2. **Path Handling Across Processes**
   - **Risk**: Inconsistent path normalization between processes could lead to file not found errors
   - **Mitigation**: Consistently use file_mgr.normalize_path() for all path operations and ensure proper absolute paths

3. **Timeline Schema Compatibility**
   - **Risk**: Generated timelines might not match schema requirements for V3 timelines
   - **Mitigation**: Add validation in YouTubeTimelineBuilder and validate all timelines against the schema

4. **Synchronization Between Processes**
   - **Risk**: File system latency could lead to race conditions
   - **Mitigation**: Use atomic file operations and implement proper existence checks with retries

### Error Handling Strategy

1. **Graceful Degradation**
   - Ensure the system continues to function even if timeline generation fails
   - Return different results based on success/failure of timeline generation

2. **Comprehensive Logging**
   - Add detailed logging throughout the timeline builder
   - Log all file operations, especially path normalization and serialization

3. **Custom Exceptions**
   - Use specialized exceptions (YouTubeTimelineError) for clear error identification
   - Wrap lower-level exceptions with context-specific information

## Implementation Sequence

1. Create youtube_timeline_builder.py with core classes
2. Add unit tests for YouTubeTimelineBuilder
3. Update yt_clips.py ProcessingOptions and process_video function
4. Add command line interface updates to yt_clips.py
5. Create integration tests
6. Add documentation and examples
7. Implement comprehensive error handling

## Success Criteria

- YouTubeTimelineBuilder can create valid V3 timelines from YouTube clips
- Timelines include proper metadata for merging in Phase 2
- The system handles channel-specific paths properly
- Command line interface supports timeline generation options
- Unit and integration tests pass
- Documentation is comprehensive and includes examples