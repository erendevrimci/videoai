# Action Plan for Asynchronous Integration of yt_clips.py

## Overview

This action plan outlines the steps to develop an asynchronous integration for `yt_clips.py` with the main VideoAI pipeline. The integration will use a timeline-based architecture where both processes operate independently and communicate through timeline files.

## Goals

1. Make `yt_clips.py` run as a completely independent, parallel process
2. Implement timeline generation in `yt_clips.py` for YouTube clips
3. Create a timeline merging component to combine outputs
4. Support multiple merging strategies for different creative needs
5. Integrate with existing timeline system and export formats

## Phase 1: Timeline Builder for YouTube Clips

### 1. Enhance yt_clips.py with Timeline Generation

- [ ] Create `YouTubeTimelineBuilder` class to convert clips to v3 timeline format
- [ ] Add methods to export timeline to JSON file with proper metadata
- [ ] Update `yt_clips.py` main function to support timeline mode
- [ ] Add command-line options for timeline operations
- [ ] Ensure proper integration with `TimelineManager` and `FileManager`

### 2. Implement Caption Integration

- [ ] Add support for captions in the timeline as TlText elements
- [ ] Create timestamp conversion utilities for caption synchronization
- [ ] Ensure caption posit

### 3. Add Channel-Specific Path Handling

- [ ] Update path handling to use `FileManager` for all operations
- [ ] Add channel number support to all relevant functions
- [ ] Implement channel-specific output directories for downloads and timelines
- [ ] Ensure consistent path normalization

## Phase 2: Timeline Merger Implementation

### 1. Create TimelineMerger Class

- [ ] Implement `TimelineMerger` class with timeline loading and merging
- [ ] Add support for various merging strategies:
  - [ ] Sequential (main content followed by clips)
  - [ ] Interleaved (alternating between main and clips)
  - [ ] Picture-in-Picture (clips as insets over main content)
- [ ] Implement proper metadata merging and preservation

### 2. Develop Timeline Export Functions

- [ ] Add functionality to export merged timeline to JSON
- [ ] Implement export to FCPXML and XML formats
- [ ] Ensure proper metadata is included in all formats
- [ ] Add validation to ensure output is compatible with target applications

### 3. Create File Monitoring System

- [ ] Implement `monitor_and_merge_timelines` function
- [ ] Add file existence detection with configurable timeout
- [ ] Create logging and error handling for monitoring
- [ ] Implement automatic triggering of timeline merging

## Phase 3: Integration and Testing

### 1. Update Main Pipeline for Timeline Output

- [ ] Ensure main pipeline creates a standard timeline file
- [ ] Add configuration options for timeline output
- [ ] Update file paths to be consistent with the monitoring system

### 2. Add Process Management Functions

- [ ] Create functions to launch `yt_clips.py` as a separate process
- [ ] Add status tracking for monitoring process completion
- [ ] Implement proper error handling and logging
- [ ] Add timeout and resource management

### 3. Implement End-to-End Testing

- [ ] Create test cases for the full workflow
- [ ] Test multiple merging strategies
- [ ] Verify exported format compatibility with target applications
- [ ] Measure performance and optimize as needed

## Phase 4: Documentation and Usability

### 1. Update Documentation

- [ ] Document command-line options for timeline mode
- [ ] Create usage examples and tutorials
- [ ] Update existing documentation to include new workflow
- [ ] Document merging strategies and their use cases

### 2. Improve Error Handling and Reporting

- [ ] Add comprehensive error messages
- [ ] Create detailed logs for troubleshooting
- [ ] Implement graceful fallbacks for common failure cases
- [ ] Add validation and sanity checks

### 3. Optimize Performance

- [ ] Identify and address performance bottlenecks
- [ ] Optimize file operations and memory usage
- [ ] Add performance monitoring for all operations
- [ ] Implement caching where appropriate

## Timeline Considerations

- Both the main pipeline and `yt_clips.py` need to produce compatible v3 timeline files
- Timeline files must include all necessary metadata for proper merging
- File paths must be properly normalized for compatibility between processes
- The monitoring system must handle various failure cases gracefully
- The merging process should preserve all metadata from both timelines

## Technical Implementation Details

### YouTubeTimelineBuilder Class

```python
class YouTubeTimelineBuilder:
    """
    Creates timeline files from YouTube clips.
    """
    def __init__(self, timeline_manager: Optional['TimelineManager'] = None):
        """Initialize with an optional timeline manager."""
        self.timeline_manager = timeline_manager or TimelineManager()
        
    def create_timeline_from_clips(self, metadata: VideoMetadata, width: int = 1920, height: int = 1080, framerate: float = 30.0) -> 'v3':
        """
        Convert processed YouTube clips to a v3 timeline.
        
        Args:
            metadata: VideoMetadata object containing clip information
            width: Output video width
            height: Output video height
            framerate: Output framerate
            
        Returns:
            v3 timeline object
        """
        # Create empty timeline with appropriate dimensions
        timeline = self.timeline_manager.create_v3_timeline(
            width=width,
            height=height,
            framerate=framerate
        )
        
        # Add clips to timeline with proper timing
        current_frame = 0
        
        for clip_idx, clip in enumerate(metadata.clips):
            clip_path = clip.path
            duration_frames = int(clip.duration * framerate)
            
            # Create TlVideo object for this clip
            video_obj = TlVideo(
                start=current_frame,
                dur=duration_frames,
                src=initFileInfo(str(clip_path)),
                offset=0,
                speed=1.0,
                stream=0
            )
            
            # Add to the first video track
            timeline.v[0].append(video_obj)
            
            # Add caption segments as text elements on the second track
            if len(timeline.v) < 2:
                timeline.v.append([])  # Add second track for captions
                
            for caption in clip.captions:
                caption_start = current_frame + int((caption.start - clip.start_time) * framerate)
                caption_end = current_frame + int((caption.end - clip.start_time) * framerate)
                caption_duration = caption_end - caption_start
                
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
            
            # Update current position
            current_frame += duration_frames
        
        # Add metadata about the source video
        timeline.videoai_metadata = {
            'version': '1.0',
            'type': 'v3',
            'created_at': datetime.now().isoformat(),
            'description': f'YouTube clips timeline for {metadata.video.title}',
            'youtube_info': {
                'video_id': metadata.video.id,
                'title': metadata.video.title,
                'url': str(metadata.video.url),
                'clip_count': len(metadata.clips)
            }
        }
        
        return timeline
```

### TimelineMerger Class

```python
class TimelineMerger:
    """
    Merges timelines from the main pipeline and YouTube clips.
    """
    def __init__(self, timeline_manager: Optional['TimelineManager'] = None):
        """Initialize with an optional timeline manager."""
        self.timeline_manager = timeline_manager or TimelineManager()
        
    def merge_timelines(self, main_timeline_path: Union[str, Path], clips_timeline_path: Union[str, Path], merge_strategy: str = "sequential") -> 'v3':
        """
        Merge two timelines using the specified strategy.
        
        Args:
            main_timeline_path: Path to the main pipeline timeline
            clips_timeline_path: Path to the YouTube clips timeline
            merge_strategy: Merging strategy to use ("sequential", "interleaved", "pip")
            
        Returns:
            Merged v3 timeline
        """
        # Load both timelines
        main_timeline = self.timeline_manager.deserialize_timeline(main_timeline_path)
        clips_timeline = self.timeline_manager.deserialize_timeline(clips_timeline_path)
        
        if not main_timeline or not clips_timeline:
            raise ValueError("Failed to load one or both timelines")
            
        # Create a new timeline with the same parameters as the main timeline
        merged_timeline = self.timeline_manager.create_v3_timeline(
            width=main_timeline.res[0],
            height=main_timeline.res[1],
            framerate=main_timeline.tb
        )
        
        # Apply the requested merge strategy
        if merge_strategy == "sequential":
            return self._merge_sequential(merged_timeline, main_timeline, clips_timeline)
        elif merge_strategy == "interleaved":
            return self._merge_interleaved(merged_timeline, main_timeline, clips_timeline)
        elif merge_strategy == "pip":
            return self._merge_pip(merged_timeline, main_timeline, clips_timeline)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
```

### Timeline Monitoring Function

```python
def monitor_and_merge_timelines(channel_number: int, timeout_seconds: int = 3600) -> bool:
    """
    Monitor for both main and clips timelines and merge them when available.
    
    Args:
        channel_number: Channel number to monitor
        timeout_seconds: Maximum time to wait for both timelines
        
    Returns:
        True if timelines were successfully merged, False otherwise
    """
    file_mgr = FileManager()
    timeline_mgr = TimelineManager(channel_number=channel_number)
    
    # Define expected timeline paths
    main_timeline_path = file_mgr.get_timeline_path("main_timeline", channel_number)
    clips_timeline_path = file_mgr.get_timeline_path("clips_timeline", channel_number)
    merged_timeline_path = file_mgr.get_timeline_path("merged_timeline", channel_number)
    
    logger.info(f"Monitoring for timeline files (channel {channel_number}):")
    logger.info(f"  Main timeline: {main_timeline_path}")
    logger.info(f"  Clips timeline: {clips_timeline_path}")
    
    start_time = time.time()
    main_ready = False
    clips_ready = False
    
    # Monitor loop
    while time.time() - start_time < timeout_seconds:
        # Check if main timeline file exists
        if not main_ready and main_timeline_path.exists():
            logger.info(f"Main timeline file detected: {main_timeline_path}")
            main_ready = True
            
        # Check if clips timeline file exists
        if not clips_ready and clips_timeline_path.exists():
            logger.info(f"Clips timeline file detected: {clips_timeline_path}")
            clips_ready = True
            
        # If both files are ready, merge them
        if main_ready and clips_ready:
            logger.info("Both timeline files are ready. Merging...")
            
            try:
                # Create merger
                merger = TimelineMerger(timeline_mgr)
                
                # Merge timelines
                merged_timeline = merger.merge_timelines(
                    main_timeline_path=main_timeline_path,
                    clips_timeline_path=clips_timeline_path,
                    merge_strategy="sequential"  # Can be configurable
                )
                
                # Export merged timeline
                merger.export_merged_timeline(
                    timeline=merged_timeline,
                    output_path=merged_timeline_path,
                    channel_number=channel_number
                )
                
                logger.info(f"Successfully merged timelines: {merged_timeline_path}")
                
                # Convert to export formats
                exports_dir = file_mgr.get_channel_output_path(channel_number) / "exports"
                format_paths = merger.convert_to_formats(
                    timeline=merged_timeline,
                    output_dir=exports_dir,
                    formats=["fcp11", "fcp7"],
                    channel_number=channel_number
                )
                
                for format_type, path in format_paths.items():
                    logger.info(f"Exported to {format_type}: {path}")
                
                return True
            except Exception as e:
                logger.error(f"Error merging timelines: {e}", exc_info=True)
                return False
        
        # If not both ready, wait briefly before checking again
        time.sleep(5)
    
    # Timeout reached
    logger.warning(f"Timeout reached waiting for timeline files (after {timeout_seconds} seconds)")
    return False
```

## Next Steps

After completing the implementation according to this action plan, we should consider:

1. Exploring advanced timeline merging strategies beyond the initial three
2. Creating a graphical tool for visualizing the merged timeline
3. Adding more configuration options for the monitoring system
4. Implementing automated testing for the entire workflow
5. Adding support for additional export formats