"""
YouTube video downloading and processing module.

This module provides functionality for searching, downloading, processing,
and analyzing YouTube videos, breaking them into scene-based clips, and
generating captions for each clip.
"""
import os
import json
import yt_dlp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, HttpUrl, Field, field_validator, model_validator
from youtube_search import YoutubeSearch
from moviepy.editor import VideoFileClip
import whisper
from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

# Import logging system components
from logging_system.logger import Logger
from logging_system.exception_handler import log_exceptions, VideoAIException

# Initialize module logger
logger = Logger.get_logger("yt_clips")


class YouTubeError(VideoAIException):
    """Base exception for all YouTube-related errors in this module."""
    pass


class YouTubeSearchError(YouTubeError):
    """Exception raised when YouTube search fails."""
    pass


class YouTubeDownloadError(YouTubeError):
    """Exception raised when YouTube video download fails."""
    pass


class SceneDetectionError(YouTubeError):
    """Exception raised when scene detection fails."""
    pass


class VideoProcessingError(YouTubeError):
    """Exception raised when video processing or splitting fails."""
    pass


class CaptionGenerationError(YouTubeError):
    """Exception raised when caption generation fails."""
    pass


class WhisperModelSize(str, Enum):
    """Available sizes for Whisper models."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class VideoSearchResult(BaseModel):
    """Model for YouTube video search results."""
    id: str
    title: str
    url: HttpUrl
    duration: str
    views: str
    thumbnail: str
    
    @field_validator('url')
    @classmethod
    def validate_youtube_url(cls, v):
        """Ensure URL is a valid YouTube URL."""
        if 'youtube.com/watch' not in str(v) and 'youtu.be/' not in str(v):
            raise ValueError(f"Not a valid YouTube URL: {v}")
        return v
    
    @classmethod
    def from_youtube_search(cls, video_data: Dict[str, Any]) -> 'VideoSearchResult':
        """Create a VideoSearchResult from YouTube search API data."""
        return cls(
            id=video_data['id'],
            title=video_data['title'],
            url=f"https://www.youtube.com/watch?v={video_data['id']}",
            duration=video_data['duration'],
            views=video_data['views'],
            thumbnail=video_data['thumbnails'][0]
        )


class VideoInfo(BaseModel):
    """Model for downloaded video metadata."""
    video_id: str
    title: str
    length: int = 0
    author: str = ""
    description: str = ""
    upload_date: Optional[str] = None
    
    @field_validator('length')
    @classmethod
    def validate_length(cls, v):
        """Ensure video length is non-negative."""
        if v < 0:
            raise ValueError("Video length cannot be negative")
        return v


class CaptionSegment(BaseModel):
    """Model for a single caption segment."""
    start: float
    end: float
    text: str
    
    @field_validator('end')
    @classmethod
    def validate_time_order(cls, v, info):
        """Ensure end time is after start time."""
        if 'start' in info.data and v < info.data['start']:
            raise ValueError(f"End time ({v}) must be greater than start time ({info.data['start']})")
        return v
    
    @field_validator('start', 'end')
    @classmethod
    def validate_positive_time(cls, v):
        """Ensure times are non-negative."""
        if v < 0:
            raise ValueError(f"Time values must be non-negative (got {v})")
        return v


class VideoClip(BaseModel):
    """Model for a processed video clip."""
    path: Path
    start_time: float
    end_time: float
    duration: float = 0.0
    captions: List[CaptionSegment] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def calculate_duration(self) -> 'VideoClip':
        """Calculate duration if not explicitly provided."""
        if self.duration == 0.0 and hasattr(self, 'start_time') and hasattr(self, 'end_time'):
            self.duration = self.end_time - self.start_time
        return self
    
    @field_validator('path')
    @classmethod
    def validate_path_exists(cls, v):
        """Ensure the clip file exists."""
        if not os.path.exists(v):
            logger.warning(f"Clip file does not exist: {v}")
        return v
    
    model_config = {
        "json_encoders": {
            Path: str  # Convert Path to string for JSON serialization
        }
    }


class VideoMetadata(BaseModel):
    """Complete metadata for a processed YouTube video."""
    video: VideoSearchResult
    video_info: VideoInfo
    download_date: datetime = Field(default_factory=datetime.now)
    clips: List[VideoClip] = Field(default_factory=list)
    processor_version: str = "1.0.0"
    
    def total_clips_duration(self) -> float:
        """Calculate the total duration of all clips."""
        return sum(clip.duration for clip in self.clips)
    
    def total_caption_segments(self) -> int:
        """Count the total number of caption segments across all clips."""
        return sum(len(clip.captions) for clip in self.clips)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Path: str
        }
    }


class ProcessingOptions(BaseModel):
    """Options for video processing."""
    max_results: int = 5
    output_directory: Path = Field(default=Path("output"))
    download_directory: Path = Field(default=Path("downloads"))
    scene_threshold: float = 15.0  # Lower default threshold to detect more scenes
    whisper_model: WhisperModelSize = WhisperModelSize.BASE
    min_clip_duration: float = 1.0
    skip_captions: bool = False
    
    # Timeline generation options
    generate_timeline: bool = True
    timeline_name: Optional[str] = None
    timeline_width: int = 1920
    timeline_height: int = 1080
    timeline_framerate: float = 30.0
    channel_number: Optional[int] = None
    
    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v):
        """Ensure max_results is positive."""
        if v <= 0:
            raise ValueError("max_results must be positive")
        return v
    
    @field_validator('scene_threshold')
    @classmethod
    def validate_threshold(cls, v):
        """Validate scene detection threshold."""
        if v <= 0:
            raise ValueError("scene_threshold must be positive")
        return v
    
    @field_validator('output_directory', 'download_directory')
    @classmethod
    def ensure_directory(cls, v):
        """Ensure directories exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @field_validator('timeline_width', 'timeline_height')
    @classmethod
    def validate_resolution(cls, v):
        """Validate timeline resolution."""
        if v <= 0:
            raise ValueError(f"Resolution dimensions must be positive")
        return v
    
    @field_validator('timeline_framerate')
    @classmethod
    def validate_framerate(cls, v):
        """Validate timeline framerate."""
        if v <= 0:
            raise ValueError("Framerate must be positive")
        return v
    
    model_config = {
        "json_encoders": {
            Path: str,
            WhisperModelSize: str
        }
    }


@log_exceptions(logger_instance=logger)
def search_youtube(query: str, max_results: int = 5) -> List[VideoSearchResult]:
    """
    Search YouTube for videos matching the query.
    
    Args:
        query: The search query string to use for YouTube search
        max_results: Maximum number of results to return
        
    Returns:
        List of VideoSearchResult objects containing video information
        
    Raises:
        YouTubeSearchError: If the search fails for any reason
    """
    logger.info(f"Searching YouTube for: '{query}' (max results: {max_results})")
    
    try:
        results = YoutubeSearch(query, max_results=max_results).to_dict()
        videos = [VideoSearchResult.from_youtube_search(video) for video in results]
        
        logger.debug(f"Found {len(videos)} videos matching query '{query}'")
        return videos
    except Exception as e:
        logger.error(f"Failed to search YouTube for '{query}'", exc_info=True)
        raise YouTubeSearchError(f"Failed to search YouTube for '{query}'", cause=e)


@log_exceptions(logger_instance=logger)
def download_video(video_url: Union[str, HttpUrl], output_path: Union[str, Path] = "downloads") -> Tuple[Optional[Path], Optional[VideoInfo]]:
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        video_url: The URL of the YouTube video to download
        output_path: Directory to save the downloaded video
        
    Returns:
        Tuple containing:
        - Path to the downloaded video file or None if download failed
        - VideoInfo object with metadata or None if download failed
          
    Raises:
        YouTubeDownloadError: If the download fails for any reason
    """
    logger.info(f"Downloading video from: {video_url}")
    
    try:
        # Ensure output path exists
        output_path = Path(output_path)
        if not output_path.exists():
            logger.debug(f"Creating output directory: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract video ID from URL
        video_url_str = str(video_url)
        if 'v=' in video_url_str:
            video_id = video_url_str.split('v=')[-1].split('&')[0]
        elif 'youtu.be/' in video_url_str:
            video_id = video_url_str.split('youtu.be/')[-1].split('?')[0]
        else:
            logger.error(f"Could not extract video ID from URL: {video_url_str}")
            return None
            
        filename = f"{video_id}.mp4"
        output_file = output_path / filename
        logger.debug(f"Output file will be: {output_file}")
        
        # Set up yt-dlp options with improved format selection
        ydl_opts = {
            # More flexible format selection that will try multiple options
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': str(output_file),
            'quiet': True,  # Normal operation - silent
            'no_warnings': True,
            'extract_flat': False,  # Need full extraction, not just metadata
            'ignoreerrors': True  # Continue on download errors
        }
        
        # Download the video
        logger.debug("Starting download with yt-dlp")
        info = None
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(video_url_str, download=True)
                if not info:
                    logger.error("yt-dlp returned no info")
                    return None
            except Exception as e:
                logger.error(f"yt-dlp error: {str(e)}")
                # Try with a more basic format string
                ydl_opts['format'] = 'best'
                logger.debug("Retrying with simpler format: 'best'")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                    try:
                        info = ydl2.extract_info(video_url_str, download=True)
                    except Exception as e2:
                        logger.error(f"yt-dlp retry error: {str(e2)}")
                        raise
        
        # Check if file was actually downloaded
        if not os.path.exists(output_file):
            logger.error(f"Download failed - file not found at {output_file}")
            return None
            
        # Create VideoInfo object
        if info:
            video_info = VideoInfo(
                video_id=video_id,
                title=info.get('title', ''),
                length=info.get('duration', 0),
                author=info.get('uploader', ''),
                description=info.get('description', ''),
                upload_date=info.get('upload_date', None)
            )
            
            logger.info(f"Successfully downloaded video: {video_info.title} ({video_id})")
            logger.debug(f"Video duration: {video_info.length} seconds")
                
            return output_file, video_info
        else:
            logger.error("Failed to get video information")
            return None
    except Exception as e:
        logger.error(f"Failed to download video from {video_url}", exc_info=True)
        raise YouTubeDownloadError(f"Failed to download video from {video_url}", cause=e)


@log_exceptions(logger_instance=logger)
def detect_scenes(video_path: Union[str, Path], threshold: float = 30.0) -> List:
    """
    Detect scene changes in a video using PySceneDetect.
    
    Args:
        video_path: Path to the video file to analyze
        threshold: Content detection threshold value (higher = fewer scenes)
        
    Returns:
        List of scene changes as tuples of (start_frame, end_frame)
        
    Raises:
        SceneDetectionError: If scene detection fails
    """
    logger.info(f"Detecting scenes in video: {video_path} (threshold: {threshold})")
    
    try:
        video_path_str = str(video_path)
        video = VideoManager([video_path_str])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        logger.debug("Starting scene detection")
        video.start()
        scene_manager.detect_scenes(frame_source=video)
        scene_list = scene_manager.get_scene_list()
        
        video.release()
        
        scene_count = len(scene_list)
        logger.info(f"Detected {scene_count} scenes in video")
        
        if scene_count > 0:
            logger.debug(f"First scene: {scene_list[0][0].get_seconds()} to {scene_list[0][1].get_seconds()} seconds")
            logger.debug(f"Last scene: {scene_list[-1][0].get_seconds()} to {scene_list[-1][1].get_seconds()} seconds")
        
        return scene_list
    except Exception as e:
        logger.error(f"Failed to detect scenes in {video_path}", exc_info=True)
        raise SceneDetectionError(f"Failed to detect scenes in {video_path}", cause=e)


@log_exceptions(logger_instance=logger)
def split_video_by_scenes(
    video_path: Union[str, Path], 
    scene_list: Sequence, 
    min_duration: float = 1.0
) -> List[VideoClip]:
    """
    Split a video into multiple clips based on detected scenes.
    
    Args:
        video_path: Path to the video file to split
        scene_list: List of scene tuples from detect_scenes()
        min_duration: Minimum clip duration in seconds, shorter clips will be skipped
        
    Returns:
        List of VideoClip objects containing clip information
        
    Raises:
        VideoProcessingError: If splitting the video fails
    """
    logger.info(f"Splitting video into {len(scene_list)} clips: {video_path}")
    
    try:
        video_path_str = str(video_path)
        logger.debug(f"Loading video file: {video_path_str}")
        video = VideoFileClip(video_path_str)
        clips = []
        
        for i, (start, end) in enumerate(scene_list):
            start_time = start.get_seconds()
            end_time = end.get_seconds()
            duration = end_time - start_time
            
            # Skip clips that are too short
            if duration < min_duration:
                logger.debug(f"Skipping scene {i+1} (too short: {duration:.2f}s < {min_duration:.2f}s)")
                continue
                
            logger.debug(f"Processing scene {i+1}/{len(scene_list)}: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
            
            # Create subclip
            clip = video.subclip(start_time, end_time)
            
            # Create clip output filename with scene number
            clip_path = Path(f"{video_path_str[:-4]}_scene_{i:03d}.mp4")
            
            # Write clip to file
            logger.debug(f"Writing clip to: {clip_path}")
            clip.write_videofile(str(clip_path), codec="libx264", audio_codec="aac", logger=None)
            
            # Create VideoClip object
            video_clip = VideoClip(
                path=clip_path,
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
            
            clips.append(video_clip)
            logger.debug(f"Clip {i+1} complete: {clip_path}")
        
        # Close the original video to free resources
        video.close()
        logger.info(f"Successfully split video into {len(clips)} clips")
        return clips
    except Exception as e:
        logger.error(f"Failed to split video {video_path} into clips", exc_info=True)
        raise VideoProcessingError(f"Failed to split video {video_path} into clips", cause=e)


@log_exceptions(logger_instance=logger)
def generate_captions(
    video_path: Union[str, Path], 
    model_name: Union[str, WhisperModelSize] = WhisperModelSize.BASE
) -> List[CaptionSegment]:
    """
    Generate captions for a video using OpenAI's Whisper model.
    
    Args:
        video_path: Path to the video or audio file to transcribe
        model_name: Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        
    Returns:
        List of CaptionSegment objects with timing and text information
        
    Raises:
        CaptionGenerationError: If generating captions fails
    """
    logger.info(f"Generating captions for: {video_path} (model: {model_name})")
    
    try:
        # Ensure model_name is string
        model_name_str = model_name.value if isinstance(model_name, WhisperModelSize) else str(model_name)
        
        # Load the specified Whisper model
        logger.debug(f"Loading Whisper model: {model_name_str}")
        model = whisper.load_model(model_name_str)
        
        # Transcribe the audio
        logger.debug("Starting transcription")
        result = model.transcribe(str(video_path))
        
        # Convert to CaptionSegment objects
        captions = [
            CaptionSegment(
                start=segment['start'],
                end=segment['end'],
                text=segment['text']
            )
            for segment in result["segments"]
        ]
        
        # Return the segments
        segment_count = len(captions)
        logger.info(f"Generated {segment_count} caption segments")
        
        if segment_count > 0:
            total_duration = captions[-1].end - captions[0].start
            logger.debug(f"Caption duration: {total_duration:.2f}s")
            
        return captions
    except Exception as e:
        logger.error(f"Failed to generate captions for {video_path}", exc_info=True)
        raise CaptionGenerationError(f"Failed to generate captions for {video_path}", cause=e)


@log_exceptions(logger_instance=logger)
def process_video(
    query: str, 
    options: Optional[ProcessingOptions] = None
) -> Union[Tuple[Path, VideoMetadata], Tuple[Path, VideoMetadata, Path], None]:
    """
    Main function to process a YouTube video: search, download, detect scenes,
    split into clips, and generate captions.
    
    Args:
        query: YouTube search query to find videos
        options: Processing options (if None, default options are used)
        
    Returns:
        Tuple containing:
        - Path to the generated metadata JSON file
        - VideoMetadata object with all processed information
        - Path to the timeline file (if timeline generation enabled)
        Returns None if processing failed
        
    Raises:
        Various exceptions from the individual processing steps
    """
    logger.info(f"Starting video processing for query: '{query}'")
    
    # Use default options if none provided
    if options is None:
        options = ProcessingOptions()
    
    try:
        # Search for videos
        videos = search_youtube(query, max_results=options.max_results)
        if not videos:
            logger.warning(f"No videos found for query: '{query}'")
            return None
        
        video_result = videos[0]
        logger.info(f"Selected video: {video_result.title} ({video_result.id})")
        
        # Download video
        try:
            download_result = download_video(
                video_result.url, 
                output_path=options.download_directory
            )
            
            # Check if result is None or a tuple with the expected values
            if download_result is None:
                logger.error("Download returned None")
                return None
                
            video_path, video_info = download_result
            
            if not video_path or not os.path.exists(video_path):
                logger.error(f"Failed to download video or file not found at {video_path}")
                return None
        except Exception as e:
            logger.error(f"Error during video download: {str(e)}")
            return None
        
        # Detect scenes
        scene_list = detect_scenes(video_path, threshold=options.scene_threshold)
        
        # If no scenes detected, treat the entire video as one scene
        if not scene_list:
            logger.warning("No scenes detected - treating entire video as one scene")
            
            try:
                # Get video properties
                with VideoFileClip(str(video_path)) as video:
                    duration = video.duration
                    fps = video.fps if video.fps else 30.0
                    
                # Create a scene with the entire video
                from scenedetect.frame_timecode import FrameTimecode
                start_frame = FrameTimecode(timecode=0, fps=fps)
                end_frame = FrameTimecode(timecode=duration, fps=fps)
                scene_list = [(start_frame, end_frame)]
                
                logger.info(f"Created single scene for entire video (duration: {duration:.2f}s)")
            except Exception as e:
                logger.error(f"Failed to create fallback scene: {str(e)}")
                return None
        
        # Split video into clips based on scenes
        clips = split_video_by_scenes(
            video_path, 
            scene_list, 
            min_duration=options.min_clip_duration
        )
        
        # Generate metadata
        logger.debug("Creating metadata structure")
        metadata = VideoMetadata(
            video=video_result,
            video_info=video_info,
            download_date=datetime.now(),
            clips=[]  # Will populate below
        )
        
        # Process each clip
        if not options.skip_captions:
            logger.info(f"Generating captions for {len(clips)} clips")
            for i, clip in enumerate(clips):
                logger.debug(f"Processing clip {i+1}/{len(clips)}: {clip.path}")
                captions = generate_captions(clip.path, model_name=options.whisper_model)
                
                # Update clip with captions
                clip.captions = captions
                metadata.clips.append(clip)
        else:
            logger.info("Skipping caption generation")
            metadata.clips = clips
        
        # Save metadata to JSON
        output_dir = options.output_directory
        if not output_dir.exists():
            logger.debug(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / f"{video_result.id}_metadata.json"
        logger.debug(f"Writing metadata to: {json_path}")
        
        # Serialize to JSON using Pydantic V2 method
        with open(json_path, 'w', encoding='utf-8') as f:
            json_content = metadata.model_dump_json(indent=2)
            f.write(json_content)
        
        logger.info(f"Processing complete. Metadata saved to: {json_path}")
        
        # Generate timeline if requested
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
                
                # Check if timeline creation was successful
                if timeline is None:
                    logger.error("Timeline creation failed - timeline is None")
                    return json_path, metadata
                    
                # Save timeline
                timeline_path = builder.save_timeline(
                    timeline=timeline,
                    timeline_name=options.timeline_name
                )
                
                # Verify timeline was saved successfully
                if timeline_path and os.path.exists(timeline_path):
                    logger.info(f"Timeline generated and saved to: {timeline_path}")
                    # Add timeline path to return data
                    return json_path, metadata, timeline_path
                else:
                    logger.error("Timeline was not saved successfully")
                    return json_path, metadata
                    
            except Exception as e:
                logger.error(f"Error generating timeline: {e}", exc_info=True)
                # Continue with normal return even if timeline generation fails
                return json_path, metadata
        else:
            # Original return
            return json_path, metadata
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return None


def main() -> None:
    """
    Main entry point for command line usage.
    
    Prompts for a YouTube search query, processes a matching video,
    and reports the results.
    """
    # Configure logging level from environment
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download and process YouTube videos")
    parser.add_argument("--query", type=str, help="YouTube search query")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--download-dir", type=str, default="downloads", help="Download directory")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum search results")
    parser.add_argument("--threshold", type=float, default=30.0, help="Scene detection threshold")
    parser.add_argument("--model", type=str, default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--min-duration", type=float, default=1.0, 
                        help="Minimum clip duration in seconds")
    parser.add_argument("--skip-captions", action="store_true", 
                        help="Skip caption generation")
    
    # Add timeline generation arguments
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
    
    args = parser.parse_args()
    
    # Set up logging based on environment
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    from logging_system.logger import LogLevel, LoggerConfig
    log_levels = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "critical": LogLevel.CRITICAL
    }
    
    # Use specified log level or default to INFO
    level = log_levels.get(log_level, LogLevel.INFO)
    Logger.initialize(LoggerConfig(log_level=level, console_level=level))
    
    logger.info("YouTube Clip Processor starting")
    
    # Get search query from command line or prompt
    search_query = args.query
    if not search_query:
        search_query = input("Enter YouTube search query: ")
    
    # Create processing options from arguments
    options = ProcessingOptions(
        max_results=args.max_results,
        output_directory=Path(args.output_dir),
        download_directory=Path(args.download_dir),
        scene_threshold=args.threshold,
        whisper_model=WhisperModelSize(args.model),
        min_clip_duration=args.min_duration,
        skip_captions=args.skip_captions,
        generate_timeline=args.generate_timeline,
        timeline_name=args.timeline_name,
        timeline_width=args.timeline_width,
        timeline_height=args.timeline_height,
        timeline_framerate=args.timeline_framerate,
        channel_number=args.channel
    )
    
    try:
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
        else:
            logger.warning("Processing failed.")
            print("Processing failed.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"An error occurred: {e}")
    
    logger.info("YouTube Clip Processor finished")


if __name__ == "__main__":
    # Required dependencies:
    # pip install yt-dlp youtube-search moviepy whisper scenedetect pydantic
    main()