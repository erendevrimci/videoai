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
from auto_editor.utils.log import Log

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

# Initialize file manager
file_mgr = FileManager()


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
        
    @log_exceptions(logger_instance=logger, default_return=None)
    def create_timeline_from_clips(self, 
                                  metadata: VideoMetadata, 
                                  width: int = 1920, 
                                  height: int = 1080, 
                                  framerate: float = 30.0) -> Optional[v3]:
        """
        Convert processed YouTube clips to a v3 timeline.
        
        Args:
            metadata: VideoMetadata object containing clip information
            width: Output video width
            height: Output video height
            framerate: Output framerate
            
        Returns:
            v3 timeline object or None if timeline creation fails
            
        Raises:
            YouTubeTimelineError: If timeline creation fails (when not using log_exceptions)
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
            src=initFileInfo(str(norm_path), Log()),
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
        try:
            video_id = getattr(metadata.video_info, 'video_id', '')
            if not video_id and hasattr(metadata, 'video') and hasattr(metadata.video, 'id'):
                video_id = metadata.video.id
                
            title = getattr(metadata, 'title', '')
            if not title and hasattr(metadata, 'video') and hasattr(metadata.video, 'title'):
                title = metadata.video.title
                
            url = ''
            if hasattr(metadata, 'video') and hasattr(metadata.video, 'url'):
                url = str(metadata.video.url)
                
            # Create metadata with safe attribute access
            timeline.videoai_metadata = {
                'version': '1.0',
                'type': 'v3',
                'created_at': datetime.now().isoformat(),
                'description': f'YouTube clips timeline for {title}',
                'channel': self.channel_number,
                'youtube_info': {
                    'video_id': video_id,
                    'title': title,
                    'url': url,
                    'clip_count': len(metadata.clips),
                    'processor_version': getattr(metadata, 'processor_version', '1.0.0')
                }
            }
        except Exception as e:
            logger.warning(f"Error adding metadata to timeline: {e}, using minimal metadata")
            # Create minimal metadata if there's an error
            timeline.videoai_metadata = {
                'version': '1.0',
                'type': 'v3',
                'created_at': datetime.now().isoformat(),
                'description': 'YouTube clips timeline',
                'channel': self.channel_number,
                'youtube_info': {
                    'clip_count': len(getattr(metadata, 'clips', [])),
                    'processor_version': '1.0.0'
                }
            }

    @log_exceptions(logger_instance=logger)
    def save_timeline(self, 
                     timeline: Optional[v3], 
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
            # Check if timeline is None
            if timeline is None:
                raise ValueError("Cannot save None timeline - timeline creation may have failed")
                
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