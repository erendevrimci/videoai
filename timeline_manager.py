"""
Timeline Manager module for VideoAI project.

Provides timeline creation, management, serialization, and visualization capabilities
by integrating auto-editor's timeline structures with VideoAI's clip sequence format.

Features:
- Timeline creation from source videos and clip sequences
- Timeline serialization and deserialization to/from JSON
- Format conversion between v1 (simple) and v3 (advanced) timelines
- ASCII visualization of timeline structures with multiple detail levels
- Timeline summary generation and export to text files
- Channel-aware file path management
- Configurable timeline settings with channel-specific overrides

The visualization capabilities include:
- Multiple detail levels (minimal, normal, detailed)
- ASCII-based proportional representation of clips
- Time markers for better navigation
- Track-based visualization for v3 timelines
- Export to text files for sharing and documentation

Configuration:
The TimelineManager integrates with the VideoAI config system, using:
- Default timeline settings (framerate, resolution, etc.)
- Customizable visualization options (detail level, width, time markers)
- Serialization options (format, compression, backup)
- Channel-specific overrides for all settings

For more details, see docs/timeline_configuration.md
"""
import json
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from fractions import Fraction
import os
import jsonschema

# Import auto-editor components
from auto_editor.timeline import v1, v3, TlVideo, TlAudio, TlImage, TlRect
from auto_editor.ffwrapper import initFileInfo, FileInfo
from auto_editor.utils.chunks import Chunks
from auto_editor.utils.log import Log

# Define a TlText class since it's not available in auto-editor
class TlText:
    """
    Text overlay element for timelines.
    
    Represents a text element that can be placed on a timeline with
    positioning, font settings, and styling options.
    """
    
    def __init__(self, 
                 start: int,
                 dur: int, 
                 text: str,
                 x: int = 0,
                 y: int = 0,
                 font: str = "Arial",
                 font_size: int = 36,
                 color: str = "#FFFFFF",
                 bg_color: str = "",
                 align: str = "center",
                 opacity: float = 1.0,
                 style: str = "normal"):
        """
        Initialize a text element for a timeline.
        
        Args:
            start: Start frame for this text element
            dur: Duration in frames
            text: Text content to display
            x: X position (pixels from left, or percentage if ends with %)
            y: Y position (pixels from top, or percentage if ends with %)
            font: Font family to use
            font_size: Font size in pixels
            color: Text color in hex format (#FFFFFF)
            bg_color: Background color in hex format (empty for transparent)
            align: Text alignment ("left", "center", "right")
            opacity: Text opacity (0.0-1.0)
            style: Text style ("normal", "bold", "italic", "bold-italic")
        """
        self.name = "text"
        self.start = start
        self.dur = dur
        self.text = text
        self.x = x
        self.y = y
        self.font = font
        self.font_size = font_size
        self.color = color
        self.bg_color = bg_color
        self.align = align
        self.opacity = opacity
        self.style = style
    
    def as_dict(self):
        """
        Convert the text element to a dictionary for serialization.
        """
        return {
            "name": self.name,
            "start": self.start,
            "dur": self.dur,
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "font": self.font,
            "font_size": self.font_size,
            "color": self.color,
            "bg_color": self.bg_color,
            "align": self.align,
            "opacity": self.opacity,
            "style": self.style
        }

# Define classes that were removed from auto_editor
class UniformChunks:
    """A wrapper for a list of (start, end) tuples."""
    
    def __init__(self, chunks, is_range=False, inverted=False):
        self.chunks = chunks
        self.inverted = inverted
        
    def __iter__(self):
        return iter(self.chunks)

def dummy_log():
    """Create a dummy logger instance"""
    return Log(is_debug=False, quiet=True)

# Import VideoAI components
from file_manager import FileManager

# Initialize file manager
file_mgr = FileManager()

# Type definitions for better type hinting
PathLike = Union[str, Path]
ClipDict = Dict[str, Any]  # VideoAI clip dictionary format
ClipSequence = List[ClipDict]  # VideoAI clip sequence format

# JSON Schema definitions for validation
V1_TIMELINE_SCHEMA = {
    "type": "object",
    "required": ["version", "source", "chunks", "videoai_metadata"],
    "properties": {
        "version": {"type": "string", "enum": ["1"]},
        "source": {"type": "string"},
        "chunks": {"type": "object"},
        "videoai_metadata": {
            "type": "object",
            "required": ["version", "type"],
            "properties": {
                "channel": {"type": ["integer", "null"]},
                "version": {"type": "string"},
                "type": {"type": "string", "enum": ["v1"]},
                "created_at": {"type": "string"},
                "description": {"type": "string"}
            }
        }
    }
}

V3_TIMELINE_SCHEMA = {
    "type": "object",
    "required": ["version", "timebase", "samplerate", "resolution", "background", "v", "a", "videoai_metadata"],
    "properties": {
        "version": {"type": "string", "enum": ["3"]},
        "timebase": {"type": "string", "pattern": "^\\d+/\\d+$"},
        "samplerate": {"type": "integer"},
        "resolution": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2
                },
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                {
                    "type": "object",
                    "additionalProperties": True
                }
            ]
        },
        "background": {"type": "string"},
        "v": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "object"}
            }
        },
        "a": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "object"}
            }
        },
        "videoai_metadata": {
            "type": "object",
            "required": ["version", "type"],
            "properties": {
                "channel": {"type": ["integer", "null"]},
                "version": {"type": "string"},
                "type": {"type": "string", "enum": ["v3"]},
                "created_at": {"type": "string"},
                "description": {"type": "string"},
                "title_desc": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
    }
}

# Custom JSON encoder for handling special data types
class TimelineEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles special types used in timelines.
    
    Handles:
    - Path objects (converted to strings)
    - Fraction objects (converted to strings as "numerator/denominator")
    - FileInfo objects (converted to their path string)
    - Tuples (converted to lists for JSON compatibility)
    """
    def default(self, obj):
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj.resolve())
            
        # Handle Fraction objects
        if isinstance(obj, Fraction):
            return f"{obj.numerator}/{obj.denominator}"
            
        # Handle FileInfo objects
        if hasattr(obj, 'path') and isinstance(obj.path, Path):
            return str(obj.path.resolve())
            
        # Handle Tuples (especially for resolution values)
        if isinstance(obj, tuple):
            return list(obj)
            
        # Handle Chunks objects
        if isinstance(obj, Chunks) and hasattr(obj, 'chunks'):
            return {
                "type": "chunks",
                "chunks": obj.chunks,
                "inverted": getattr(obj, 'inverted', False)
            }
            
        # Let the base class handle everything else
        return super().default(obj)


class TimelineManager:
    """
    Timeline Manager for creating, manipulating, and serializing video timelines.
    
    Integrates auto-editor's timeline structures with VideoAI's clip sequence format,
    providing a bridge between the two systems and enabling advanced editing capabilities.
    """
    
    def __init__(self, channel_number: Optional[int] = None, log: Optional[Log] = None):
        """
        Initialize the TimelineManager.
        
        Args:
            channel_number: Channel number for context-specific operations
            log: Auto-editor Log object for output (creates one if None)
        """
        self.channel_number = channel_number
        self.log = log or dummy_log()
        
        # Try to import logging system for VideoAI-specific logging
        try:
            from logging_system import Logger, LogLevel
            self.logger = Logger.get_logger("timeline_manager")
            self.has_logging = True
        except ImportError:
            self.has_logging = False
            
    def _log_info(self, message: str) -> None:
        """Log an info message using the appropriate logging system."""
        if self.has_logging:
            self.logger.info(message)
        else:
            print(f"[TimelineManager] {message}")
            
    def _log_error(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """Log an error message using the appropriate logging system."""
        if self.has_logging:
            self.logger.error(message, exc_info=exc_info is not None)
        else:
            print(f"[TimelineManager ERROR] {message}")
            if exc_info:
                traceback.print_exc()
                
    def _get_timeline_path(self, timeline_name: str) -> Path:
        """
        Get the path for storing a timeline file.
        
        Args:
            timeline_name: Base name for the timeline file
            
        Returns:
            Path object for the timeline file
        """
        # Delegate to the public method for consistency
        return self.get_timeline_path(timeline_name)
        
    def _get_timeline_config(self):
        """
        Get the timeline configuration with channel-specific overrides.
        
        Returns:
            TimelineConfig object with channel-specific overrides applied
        """
        # Import here to avoid circular import at module level
        from config import get_timeline_config
        return get_timeline_config(self.channel_number)
            
    def create_v1_timeline(self, source_path: PathLike, chunks: Optional[Chunks] = None) -> v1:
        """
        Create a v1 timeline (simple chunks-based timeline).
        
        Args:
            source_path: Path to the source video file
            chunks: Optional Chunks object defining kept/cut sections
            
        Returns:
            v1 timeline object
        """
        try:
            # Normalize path
            source_path = file_mgr.normalize_path(source_path)
            
            # Initialize source FileInfo
            source = initFileInfo(str(source_path), self.log)
            
            # Create default chunks if none provided (keeps everything)
            if chunks is None:
                # Create chunks that keep the entire video
                video_duration_frames = int(source.video.duration * source.video.fps)
                chunks = UniformChunks([(0, video_duration_frames)], is_range=True)
                
            # Create the v1 timeline
            timeline = v1(source=source, chunks=chunks)
            
            self._log_info(f"Created v1 timeline for {source_path}")
            return timeline
            
        except Exception as e:
            self._log_error(f"Error creating v1 timeline: {e}", exc_info=e)
            raise
            
    def create_v3_timeline(self, 
                          source_path: Optional[PathLike] = None, 
                          width: Optional[int] = None, 
                          height: Optional[int] = None, 
                          framerate: Optional[Union[int, float, Fraction]] = None,
                          samplerate: Optional[int] = None,
                          background: Optional[str] = None) -> v3:
        """
        Create a v3 timeline (multi-track advanced timeline).
        
        Args:
            source_path: Optional path to a template source for timeline settings
            width: Output video width in pixels (if None, uses config value)
            height: Output video height in pixels (if None, uses config value)
            framerate: Output framerate (can be int, float, or Fraction)
            samplerate: Output audio sample rate (if None, uses config value)
            background: Background color in hex format (if None, uses config value)
            
        Returns:
            v3 timeline object
        """
        try:
            # Get configuration
            timeline_config = self._get_timeline_config()
            
            # Use configuration defaults for any unspecified values
            if width is None:
                width = timeline_config.default_width
                
            if height is None:
                height = timeline_config.default_height
                
            if framerate is None:
                framerate = timeline_config.default_framerate
                
            if samplerate is None:
                samplerate = timeline_config.default_samplerate
                
            if background is None:
                background = timeline_config.default_background
            
            # Convert framerate to Fraction if needed
            if not isinstance(framerate, Fraction):
                framerate = Fraction(framerate).limit_denominator(1000)
                
            # Initialize source if provided
            source = None
            if source_path is not None:
                source_path = file_mgr.normalize_path(source_path)
                source = initFileInfo(str(source_path), self.log)
                
            # Create empty v3 timeline
            timeline = v3(
                src=source,
                tb=framerate,
                sr=samplerate,
                res=(width, height),
                background=background,
                v=[[]],  # Empty video track
                a=[[]],  # Empty audio track
                v1=None  # Not v1 compatible
            )
            
            # Add required videoai_metadata with minimal fields to pass schema validation
            from datetime import datetime
            timeline.videoai_metadata = {
                'version': '1.0',
                'type': 'v3',
                'created_at': datetime.now().isoformat(),
                'description': 'Empty timeline',
                'channel': self.channel_number
            }
            
            self._log_info(f"Created v3 timeline with resolution {width}x{height}, {framerate} fps")
            return timeline
            
        except Exception as e:
            self._log_error(f"Error creating v3 timeline: {e}", exc_info=e)
            raise
            
    def clip_sequence_to_timeline(self, 
                                clip_sequence: ClipSequence, 
                                output_width: int = 1080, 
                                output_height: int = 1920, 
                                framerate: Union[int, float, Fraction] = 30,
                                clips_dir: Optional[PathLike] = None) -> v3:
        """
        Convert a VideoAI clip sequence to a v3 timeline.
        
        Args:
            clip_sequence: List of VideoAI clip dictionaries
            output_width: Output video width in pixels
            output_height: Output video height in pixels
            framerate: Output framerate
            clips_dir: Base directory for clip paths (defaults to config value)
            
        Returns:
            v3 timeline object
        """
        try:
            # Ensure we have clip paths
            if clips_dir is None:
                from config import config
                clips_dir = file_mgr.get_abs_path(config.file_paths.clips_directory)
                
            # Create empty v3 timeline
            timeline = self.create_v3_timeline(
                width=output_width,
                height=output_height,
                framerate=framerate
            )
            
            # Convert framerate to Fraction if needed
            if not isinstance(framerate, Fraction):
                framerate = Fraction(framerate).limit_denominator(1000)
                
            # Process each clip and add to timeline
            current_time = 0  # Current timeline position in frames
            
            for clip in clip_sequence:
                clip_name = clip.get('clip_name', '')
                start_time = clip.get('start_time', 0)
                duration = clip.get('duration', 0)
                
                # Calculate frame numbers
                start_frame = int(current_time)
                duration_frames = int(duration * framerate)
                
                # Find the actual clip file
                clip_path = Path(clips_dir) / clip_name
                video_file = file_mgr.find_video_file(clip_path)
                
                if not video_file:
                    self._log_error(f"Could not find video file for clip: {clip_name}")
                    continue
                
                # Initialize file info
                src = initFileInfo(str(video_file), self.log)
                
                # Calculate offset in source clip (in frames)
                offset_frames = int(start_time * framerate)
                
                # Create TlVideo object
                video_obj = TlVideo(
                    start=start_frame,
                    dur=duration_frames,
                    src=src,
                    offset=offset_frames,
                    speed=1.0,
                    stream=0
                )
                
                # Add to first video track
                timeline.v[0].append(video_obj)
                
                # Update current time
                current_time += duration_frames
                
            self._log_info(f"Converted clip sequence with {len(clip_sequence)} clips to timeline")
            return timeline
            
        except Exception as e:
            self._log_error(f"Error converting clip sequence to timeline: {e}", exc_info=e)
            raise
            
    def serialize_timeline(self, timeline: Union[v1, v3], path: Optional[PathLike] = None,
                           description: str = "", validate: Optional[bool] = None) -> Dict[str, Any]:
        """
        Serialize a timeline to JSON format.
        
        Args:
            timeline: Timeline object to serialize
            path: Optional path to save JSON file
            description: Optional description of the timeline
            validate: Whether to validate the JSON against schema (if None, uses config value)
            
        Returns:
            Dictionary representation of the timeline
        """
        try:
            # Get configuration
            timeline_config = self._get_timeline_config()
            serialization_config = timeline_config.serialization
            
            # Use configuration defaults if not specified
            if validate is None:
                validate = serialization_config.validate_schema
                
            # Convert to dictionary
            if isinstance(timeline, v1):
                timeline_dict = timeline.as_dict()
            elif isinstance(timeline, v3):
                timeline_dict = timeline.as_dict()
                
                # Specifically handle resolution if it's a tuple
                if isinstance(timeline_dict.get('resolution'), tuple):
                    timeline_dict['resolution'] = list(timeline_dict['resolution'])
            else:
                raise ValueError(f"Unsupported timeline type: {type(timeline)}")
                
            # Preserve existing metadata if present, otherwise create new
            if hasattr(timeline, 'videoai_metadata') and timeline.videoai_metadata:
                timeline_dict['videoai_metadata'] = timeline.videoai_metadata
            elif serialization_config.include_metadata:
                from datetime import datetime
                timeline_dict['videoai_metadata'] = {
                    'channel': self.channel_number,
                    'version': '1.0',
                    'type': 'v1' if isinstance(timeline, v1) else 'v3',
                    'created_at': datetime.now().isoformat(),
                    'description': description
                }
            
            # Update metadata description if provided
            if description and 'videoai_metadata' in timeline_dict:
                timeline_dict['videoai_metadata']['description'] = description
            
            # Validate against schema if configured
            if validate:
                try:
                    schema = V1_TIMELINE_SCHEMA if isinstance(timeline, v1) else V3_TIMELINE_SCHEMA
                    jsonschema.validate(instance=timeline_dict, schema=schema)
                    self._log_info("Timeline JSON validated successfully against schema")
                except jsonschema.exceptions.ValidationError as ve:
                    self._log_error(f"Timeline JSON validation error: {ve}")
                    # Continue anyway - validation is just a warning
            
            # Save to file if path provided
            if path is not None:
                file_path = file_mgr.normalize_path(path)
                file_mgr.ensure_dir_exists(file_path.parent)
                
                # Create backup if auto_backup is enabled
                if serialization_config.auto_backup and file_path.exists():
                    import shutil
                    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                    try:
                        shutil.copy2(file_path, backup_path)
                        self._log_info(f"Created backup of existing timeline at {backup_path}")
                    except Exception as be:
                        self._log_error(f"Error creating backup: {be}")
                
                # Serialize based on configured format
                if serialization_config.format.lower() == 'json':
                    # Use custom JSON encoder for proper serialization
                    json_str = json.dumps(timeline_dict, cls=TimelineEncoder, indent=2)
                    
                    # Apply compression if configured
                    if serialization_config.compression:
                        import gzip
                        import base64
                        # Compress the JSON string
                        compressed = gzip.compress(json_str.encode('utf-8'))
                        
                        # If compression is enabled, store as base64-encoded gzip data
                        file_mgr.write_binary(file_path, compressed)
                        self._log_info(f"Saved compressed timeline to {file_path}")
                    else:
                        # Regular text storage
                        file_mgr.write_text(file_path, json_str)
                        self._log_info(f"Saved timeline to {file_path}")
                        
                else:
                    # Default to JSON if format is not recognized
                    self._log_error(f"Unsupported serialization format: {serialization_config.format}, using JSON")
                    json_str = json.dumps(timeline_dict, cls=TimelineEncoder, indent=2)
                    file_mgr.write_text(file_path, json_str)
                    self._log_info(f"Saved timeline to {file_path}")
                
            return timeline_dict
            
        except Exception as e:
            self._log_error(f"Error serializing timeline: {e}", exc_info=e)
            raise
            
    def deserialize_timeline(self, path: PathLike, validate: Optional[bool] = None) -> Union[v1, v3, None]:
        """
        Deserialize a timeline from a JSON file.
        
        Args:
            path: Path to the JSON timeline file
            validate: Whether to validate the JSON against schema (if None, uses config value)
            
        Returns:
            Reconstructed timeline object or None if deserialization fails
        """
        try:
            # Get configuration
            timeline_config = self._get_timeline_config()
            serialization_config = timeline_config.serialization
            
            # Use configuration defaults if not specified
            if validate is None:
                validate = serialization_config.validate_schema
                
            # Read file
            file_path = file_mgr.normalize_path(path)
            
            # Check file format (compressed or not)
            is_compressed = False
            timeline_json = None
            
            # First try to read as text (uncompressed)
            timeline_json = file_mgr.read_text(file_path)
            
            # If text reading failed, try binary reading (compressed)
            if not timeline_json:
                try:
                    binary_data = file_mgr.read_binary(file_path)
                    if binary_data:
                        # Try to decompress
                        import gzip
                        try:
                            # Decompress binary data
                            timeline_json = gzip.decompress(binary_data).decode('utf-8')
                            is_compressed = True
                            self._log_info(f"Successfully decompressed timeline file: {file_path}")
                        except Exception as ge:
                            self._log_error(f"Failed to decompress file (not gzip format): {ge}")
                except Exception as be:
                    self._log_error(f"Failed to read binary data: {be}")
            
            if not timeline_json:
                self._log_error(f"Failed to read timeline file: {file_path}")
                return None
            
            try:
                timeline_dict = json.loads(timeline_json)
            except json.JSONDecodeError as je:
                self._log_error(f"Invalid JSON in timeline file: {je}")
                return None
                
            # Validate against schema if requested
            if validate:
                try:
                    version = timeline_dict.get('version')
                    if version == '1':
                        jsonschema.validate(instance=timeline_dict, schema=V1_TIMELINE_SCHEMA)
                    elif version == '3':
                        jsonschema.validate(instance=timeline_dict, schema=V3_TIMELINE_SCHEMA)
                    else:
                        self._log_error(f"Unknown timeline version for validation: {version}")
                except jsonschema.exceptions.ValidationError as ve:
                    self._log_error(f"Timeline JSON validation error: {ve}")
                    # Continue anyway - validation is just a warning
                
            # Restore channel number from metadata if present
            metadata = timeline_dict.get('videoai_metadata', {})
            if 'channel' in metadata and metadata['channel'] is not None:
                self.channel_number = metadata['channel']
                
            # Determine timeline type and reconstruct
            version = timeline_dict.get('version')
            
            if version == '1':
                return self._deserialize_v1_timeline(timeline_dict)
            elif version == '3':
                return self._deserialize_v3_timeline(timeline_dict)
            else:
                self._log_error(f"Unknown timeline version: {version}")
                return None
                
        except Exception as e:
            self._log_error(f"Error deserializing timeline: {e}", exc_info=e)
            return None
            
    def _deserialize_v1_timeline(self, timeline_dict: Dict[str, Any]) -> Optional[v1]:
        """
        Deserialize a v1 timeline from a dictionary.
        
        Args:
            timeline_dict: Dictionary representation of the timeline
            
        Returns:
            v1 timeline object or None if deserialization fails
        """
        try:
            # Extract required fields
            source_path = timeline_dict.get('source')
            if not source_path:
                self._log_error("Missing source path in v1 timeline")
                return None
                
            chunks_data = timeline_dict.get('chunks')
            if not chunks_data:
                self._log_error("Missing chunks data in v1 timeline")
                return None
                
            # Initialize source
            source = initFileInfo(str(source_path), self.log)
            
            # Reconstruct chunks based on format
            if isinstance(chunks_data, dict) and chunks_data.get('type') == 'chunks':
                # Our custom format
                chunks_list = chunks_data.get('chunks', [])
                is_inverted = chunks_data.get('inverted', False)
                chunks = UniformChunks(chunks_list, is_range=True)
                if is_inverted:
                    chunks.inverted = True
            else:
                # Try to use auto-editor's built-in deserialization if available
                try:
                    if hasattr(Chunks, 'from_json'):
                        chunks = Chunks.from_json(chunks_data)
                    else:
                        # Fallback: create chunks directly
                        chunks = UniformChunks(chunks_data, is_range=True)
                except Exception as ce:
                    self._log_error(f"Error reconstructing chunks: {ce}", exc_info=ce)
                    # Fallback to empty chunks
                    chunks = UniformChunks([], is_range=True)
            
            # Create v1 timeline
            timeline = v1(source=source, chunks=chunks)
            
            # Restore channel number from metadata if present
            metadata = timeline_dict.get('videoai_metadata', {})
            if 'channel' in metadata and metadata['channel'] is not None:
                self.channel_number = metadata['channel']
                
            self._log_info(f"Successfully deserialized v1 timeline from {source_path}")
            return timeline
            
        except Exception as e:
            self._log_error(f"Error deserializing v1 timeline: {e}", exc_info=e)
            return None
            
    def _deserialize_v3_timeline(self, timeline_dict: Dict[str, Any]) -> Optional[v3]:
        """
        Deserialize a v3 timeline from a dictionary.
        
        Args:
            timeline_dict: Dictionary representation of the timeline
            
        Returns:
            v3 timeline object or None if deserialization fails
        """
        try:
            # Extract required fields
            timebase_str = timeline_dict.get('timebase', '30/1')
            samplerate = timeline_dict.get('samplerate', 48000)
            resolution = timeline_dict.get('resolution', [1920, 1080])
            background = timeline_dict.get('background', '#000000')
            v_tracks = timeline_dict.get('v', [[]])
            a_tracks = timeline_dict.get('a', [[]])
            metadata = timeline_dict.get('videoai_metadata', {})
            
            # Parse timebase
            if isinstance(timebase_str, str) and '/' in timebase_str:
                num, denom = map(int, timebase_str.split('/'))
                timebase = Fraction(num, denom)
            else:
                timebase = Fraction(30, 1)  # Default to 30fps
                
            # Create reconstructed video tracks
            v_reconstructed = []
            for track in v_tracks:
                v_track = []
                for obj in track:
                    obj_type = obj.get('name')
                    
                    if obj_type == 'video':
                        # Initialize source FileInfo
                        src_path = obj.get('src')
                        if src_path and isinstance(src_path, str):
                            try:
                                src = initFileInfo(src_path, self.log)
                                
                                # Create TlVideo object
                                v_obj = TlVideo(
                                    start=obj.get('start', 0),
                                    dur=obj.get('dur', 0),
                                    src=src,
                                    offset=obj.get('offset', 0),
                                    speed=obj.get('speed', 1.0),
                                    stream=obj.get('stream', 0)
                                )
                                v_track.append(v_obj)
                            except Exception as se:
                                self._log_error(f"Error initializing source for {src_path}: {se}")
                                continue
                    
                    elif obj_type == 'image':
                        # Initialize image source
                        src_path = obj.get('src')
                        if src_path and isinstance(src_path, str):
                            try:
                                src = initFileInfo(src_path, self.log)
                                
                                # Create TlImage object
                                img_obj = TlImage(
                                    start=obj.get('start', 0),
                                    dur=obj.get('dur', 0),
                                    src=src,
                                    x=obj.get('x', 0),
                                    y=obj.get('y', 0),
                                    width=obj.get('width', 0),
                                    opacity=obj.get('opacity', 1.0)
                                )
                                v_track.append(img_obj)
                            except Exception as se:
                                self._log_error(f"Error initializing image source for {src_path}: {se}")
                                continue
                    
                    elif obj_type == 'rect':
                        # Create TlRect object
                        rect_obj = TlRect(
                            start=obj.get('start', 0),
                            dur=obj.get('dur', 0),
                            x=obj.get('x', 0),
                            y=obj.get('y', 0),
                            width=obj.get('width', 0),
                            height=obj.get('height', 0),
                            fill=obj.get('fill', '#c4c4c4')
                        )
                        v_track.append(rect_obj)
                        
                    elif obj_type == 'text':
                        # Create TlText object
                        text_obj = TlText(
                            start=obj.get('start', 0),
                            dur=obj.get('dur', 0),
                            text=obj.get('text', ''),
                            x=obj.get('x', 0),
                            y=obj.get('y', 0),
                            font=obj.get('font', 'Arial'),
                            font_size=obj.get('font_size', 36),
                            color=obj.get('color', '#FFFFFF'),
                            bg_color=obj.get('bg_color', ''),
                            align=obj.get('align', 'center'),
                            opacity=obj.get('opacity', 1.0),
                            style=obj.get('style', 'normal')
                        )
                        v_track.append(text_obj)
                
                v_reconstructed.append(v_track)
            
            # Create reconstructed audio tracks
            a_reconstructed = []
            for track in a_tracks:
                a_track = []
                for obj in track:
                    if obj.get('name') == 'audio':
                        # Initialize source FileInfo
                        src_path = obj.get('src')
                        if src_path and isinstance(src_path, str):
                            try:
                                src = initFileInfo(src_path, self.log)
                                
                                # Create TlAudio object
                                a_obj = TlAudio(
                                    start=obj.get('start', 0),
                                    dur=obj.get('dur', 0),
                                    src=src,
                                    offset=obj.get('offset', 0),
                                    speed=obj.get('speed', 1.0),
                                    volume=obj.get('volume', 1.0),
                                    stream=obj.get('stream', 0)
                                )
                                a_track.append(a_obj)
                            except Exception as se:
                                self._log_error(f"Error initializing audio source for {src_path}: {se}")
                                continue
                
                a_reconstructed.append(a_track)
            
            # Create v3 timeline
            # Initialize source if available (for template settings)
            src = None
            if len(v_reconstructed) > 0 and len(v_reconstructed[0]) > 0:
                first_obj = v_reconstructed[0][0]
                if hasattr(first_obj, 'src'):
                    src = first_obj.src
                    
            # Create timeline with reconstructed tracks
            # Ensure resolution is properly converted to tuple
            if isinstance(resolution, list) and len(resolution) == 2:
                resolution_tuple = tuple(int(x) if isinstance(x, (int, float)) else x for x in resolution)
            else:
                # Default resolution if invalid format
                resolution_tuple = (1920, 1080)
                self._log_error(f"Invalid resolution format: {resolution}, using default 1920x1080")
                
            timeline = v3(
                src=src,
                tb=timebase,
                sr=samplerate,
                res=resolution_tuple,
                background=background,
                v=v_reconstructed,
                a=a_reconstructed,
                v1=None  # Not v1 compatible by default
            )
            
            # Add the metadata back to the timeline if available
            if metadata:
                timeline.videoai_metadata = metadata
            
            # Restore channel number from metadata if present
            if 'channel' in metadata and metadata['channel'] is not None:
                self.channel_number = metadata['channel']
                
            self._log_info(f"Successfully deserialized v3 timeline with {len(v_reconstructed)} video tracks and {len(a_reconstructed)} audio tracks")
            return timeline
            
        except Exception as e:
            self._log_error(f"Error deserializing v3 timeline: {e}", exc_info=e)
            return None
            
    def visualize_timeline(self, timeline: Union[v1, v3], 
                            width: Optional[int] = None, 
                            detail_level: Optional[str] = None) -> str:
        """
        Create an ASCII visualization of a timeline.
        
        Args:
            timeline: Timeline object to visualize
            width: Width of the visualization in characters (if None, uses config value)
            detail_level: Level of detail for visualization ('minimal', 'normal', 'detailed')
                         (if None, uses config value)
            
        Returns:
            ASCII string visualization of the timeline
        """
        try:
            # Get configuration
            timeline_config = self._get_timeline_config()
            viz_config = timeline_config.visualization
            
            # Use configuration defaults if not specified
            if width is None:
                width = viz_config.max_width
                
            if detail_level is None:
                detail_level = viz_config.default_detail_level
                
            # Validate detail level
            valid_levels = ['minimal', 'normal', 'detailed']
            if detail_level not in valid_levels:
                self._log_error(f"Invalid detail level: {detail_level}, using 'normal'")
                detail_level = 'normal'
                
            # Create visualization based on timeline type
            if isinstance(timeline, v1):
                return self._visualize_v1(timeline, width, detail_level)
            elif isinstance(timeline, v3):
                return self._visualize_v3(timeline, width, detail_level)
            else:
                raise ValueError(f"Unsupported timeline type: {type(timeline)}")
                
        except Exception as e:
            self._log_error(f"Error visualizing timeline: {e}", exc_info=e)
            return f"Error visualizing timeline: {str(e)}"
    
    def _create_time_markers(self, width: int, total_duration: int, fps: float = 30.0, markers: Optional[int] = None) -> str:
        """
        Create time markers for timeline visualization.
        
        Args:
            width: Width of visualization in characters
            total_duration: Total duration in frames
            fps: Frames per second for time conversion
            markers: Number of time markers to create (if None, uses config value)
            
        Returns:
            String with evenly spaced time markers
        """
        # Get configuration if markers not specified
        if markers is None:
            timeline_config = self._get_timeline_config()
            markers = timeline_config.visualization.time_markers
        
        marker_line = "|"
        labels_line = "0s"
        
        if markers < 2:
            markers = 2
            
        # Calculate marker positions
        for i in range(1, markers):
            position = (width - 1) * i // (markers - 1)
            frame = total_duration * i // (markers - 1)
            time_sec = frame / fps
            
            # Add marker
            marker_line = marker_line[:position] + "|" + marker_line[position+1:]
            
            # Add time label (right-aligned to the marker)
            label = f"{time_sec:.1f}s"
            label_pos = max(0, position - len(label) // 2)
            while len(labels_line) <= label_pos:
                labels_line += " "
            labels_line = labels_line[:label_pos] + label + labels_line[label_pos+len(label):]
        
        return marker_line + "\n" + labels_line
            
    def _visualize_v1(self, timeline: v1, width: int = 80, detail_level: str = 'normal') -> str:
        """
        Create an ASCII visualization of a v1 timeline.
        
        Args:
            timeline: v1 timeline object to visualize
            width: Width of the visualization in characters
            detail_level: Level of detail for visualization ('minimal', 'normal', 'detailed')
            
        Returns:
            ASCII string visualization of the timeline
        """
        chunks = timeline.chunks
        source = timeline.source
        
        # Get video duration in frames
        if hasattr(source, 'video') and source.video:
            total_frames = int(source.video.duration * source.video.fps)
            fps = source.video.fps
        else:
            total_frames = 1000  # Fallback
            fps = 30.0  # Fallback
        
        # Start with header based on detail level
        if detail_level == 'minimal':
            result = f"Timeline: {source.path.name} ({total_frames} frames)\n"
        else:
            result = f"Timeline visualization for: {source.path.name}\n"
            result += f"Duration: {total_frames} frames, "
            if hasattr(source, 'video') and source.video:
                result += f"{source.video.duration:.2f} seconds @ {source.video.fps} fps\n"
            else:
                result += f"Unknown duration\n"
        
        # Add metadata for detailed view
        if detail_level == 'detailed':
            result += "\nSource details:\n"
            if hasattr(source, 'video') and source.video:
                result += f"  - Resolution: {source.video.width}x{source.video.height}\n"
                result += f"  - Codec: {getattr(source.video, 'codec', 'Unknown')}\n"
            if hasattr(source, 'audio') and source.audio:
                result += f"  - Audio: {getattr(source.audio, 'samplerate', 'Unknown')} Hz, "
                result += f"{getattr(source.audio, 'channels', 'Unknown')} channels\n"
        
        # Create visual representation of chunks
        result += "\nChunks:\n"
        
        # Determine if chunks are inverted
        is_inverted = getattr(chunks, 'inverted', False)
        result += f"Mode: {'Cut segments' if is_inverted else 'Keep segments'}\n"
        
        # Draw time markers
        result += "\n" + "-" * width + "\n"
        if detail_level != 'minimal':
            result += self._create_time_markers(width, total_frames, fps) + "\n"
        
        # Draw visualization based on detail level
        if hasattr(chunks, 'chunks'):
            for i, chunk in enumerate(chunks.chunks):
                start, end = chunk[:2]  # First two elements are start and end
                
                # Calculate positions
                start_pos = int(start / total_frames * (width - 1))
                end_pos = int(end / total_frames * (width - 1))
                
                # Ensure minimum width for visibility
                if end_pos <= start_pos:
                    end_pos = start_pos + 1
                
                # Create visualization line
                if detail_level == 'minimal':
                    # Simple visualization
                    line = " " * start_pos + "#" * (end_pos - start_pos) + " " * (width - end_pos)
                    result += f"{i+1:2d} | {line}\n"
                elif detail_level == 'normal':
                    # Add timestamps
                    line = " " * start_pos + "#" * (end_pos - start_pos) + " " * (width - end_pos)
                    result += f"{i+1:2d} | {line} | {start}-{end}\n"
                else:  # detailed
                    # Add detailed information
                    line = " " * start_pos + "#" * (end_pos - start_pos) + " " * (width - end_pos)
                    start_time = start / fps
                    end_time = end / fps
                    duration = end_time - start_time
                    result += f"{i+1:2d} | {line} | Frames: {start}-{end} | Time: {start_time:.2f}s-{end_time:.2f}s | Duration: {duration:.2f}s\n"
                
                # Add small gap between chunks for detailed view
                if detail_level == 'detailed' and i < len(chunks.chunks) - 1:
                    result += "   |\n"
        
        result += "-" * width + "\n"
        
        # Add summary for detailed view
        if detail_level == 'detailed':
            total_kept_frames = sum(chunk[1] - chunk[0] for chunk in chunks.chunks)
            if is_inverted:
                kept_frames = total_frames - total_kept_frames
            else:
                kept_frames = total_kept_frames
            percent_kept = (kept_frames / total_frames) * 100 if total_frames > 0 else 0
            
            result += f"\nSummary:\n"
            result += f"  - Total timeline frames: {total_frames}\n"
            result += f"  - Kept frames: {kept_frames} ({percent_kept:.1f}%)\n"
            result += f"  - Cut frames: {total_frames - kept_frames} ({100 - percent_kept:.1f}%)\n"
        
        return result
    
    def _visualize_v3(self, timeline: v3, width: int = 80, detail_level: str = 'normal') -> str:
        """
        Create an ASCII visualization of a v3 timeline.
        
        Args:
            timeline: v3 timeline object to visualize
            width: Width of the visualization in characters
            detail_level: Level of detail for visualization ('minimal', 'normal', 'detailed')
            
        Returns:
            ASCII string visualization of the timeline
        """
        # Calculate total duration and other properties
        total_frames = timeline.end
        fps = float(timeline.tb) if timeline.tb else 30.0
        total_seconds = total_frames / fps if fps else 0
        
        # Start with header
        if detail_level == 'minimal':
            result = f"Timeline v3: {total_frames} frames ({total_seconds:.2f}s)\n"
        else:
            result = f"Timeline v3 visualization\n"
            result += f"Duration: {total_frames} frames, {total_seconds:.2f} seconds @ {fps} fps\n"
            result += f"Resolution: {timeline.res[0]}x{timeline.res[1]}, Samplerate: {timeline.sr} Hz\n"
            
            # Add title/description if available in metadata
            if hasattr(timeline, 'videoai_metadata') and timeline.videoai_metadata:
                if 'title_desc' in timeline.videoai_metadata:
                    title_desc = timeline.videoai_metadata['title_desc']
                    if 'title' in title_desc:
                        result += f"\nTitle: {title_desc['title']}\n"
                    if 'description' in title_desc and detail_level == 'detailed':
                        desc = title_desc['description']
                        # Truncate long descriptions for normal detail level
                        if detail_level == 'normal' and len(desc) > 100:
                            desc = desc[:97] + "..."
                        result += f"Description: {desc}\n"
                    if 'tags' in title_desc and detail_level == 'detailed':
                        tags = title_desc['tags']
                        if isinstance(tags, list):
                            result += f"Tags: {', '.join(tags)}\n"
        
        # Add sources for detailed view
        if detail_level == 'detailed':
            result += "\nSources:\n"
            
            # Safely get unique sources, handling None sources
            try:
                unique_sources = []
                for source in timeline.sources:
                    if source is not None and hasattr(source, 'path'):
                        if not any(s.path == source.path for s in unique_sources if hasattr(s, 'path')):
                            unique_sources.append(source)
                
                if not unique_sources:
                    result += "  No media sources\n"
                else:
                    for i, source in enumerate(unique_sources):
                        result += f"  {i+1}. {source.path.name}"
                        if hasattr(source, 'video') and source.video:
                            result += f" ({source.video.width}x{source.video.height}, {source.video.duration:.2f}s)"
                        result += "\n"
            except Exception as e:
                self._log_error(f"Error listing sources: {e}")
                result += "  Error listing sources\n"
        
        # Draw time marker scale
        result += "\n" + "-" * width + "\n"
        if detail_level != 'minimal':
            result += self._create_time_markers(width, total_frames, fps) + "\n"
        
        # Video tracks
        result += "\nVideo Tracks:\n"
        if not timeline.v or all(not track for track in timeline.v):
            result += "  No video clips\n"
        else:
            for track_idx, track in enumerate(timeline.v):
                if not track:
                    continue
                    
                # Track header
                result += f"Track {track_idx+1}:\n"
                
                # Create timeline canvas for this track
                canvas = [" " * width for _ in range(2)]  # 2 lines per track
                
                # Place clips on the canvas
                for i, obj in enumerate(track):
                    # Calculate positions
                    start_pos = int((obj.start / total_frames) * (width - 1)) if total_frames > 0 else 0
                    end_pos = int(((obj.start + obj.dur) / total_frames) * (width - 1)) if total_frames > 0 else width
                    
                    # Ensure minimum width for visibility
                    if end_pos <= start_pos:
                        end_pos = start_pos + 1
                    end_pos = min(end_pos, width)
                    
                    # Draw clip on canvas
                    clip_char = "="  # Normal video clip
                    if isinstance(obj, TlImage):
                        clip_char = "I"  # Image
                    elif isinstance(obj, TlRect):
                        clip_char = "R"  # Rectangle
                    elif isinstance(obj, TlText):
                        clip_char = "T"  # Text
                    
                    # Draw top line (clip boundary)
                    canvas[0] = canvas[0][:start_pos] + "+" + clip_char * (end_pos - start_pos - 2) + "+" + canvas[0][end_pos:]
                    
                    # Draw bottom line (clip index)
                    clip_label = f"{i+1}"
                    label_pos = start_pos + (end_pos - start_pos) // 2 - len(clip_label) // 2
                    label_pos = max(start_pos, min(label_pos, end_pos - len(clip_label)))
                    
                    if end_pos - start_pos > len(clip_label):
                        canvas[1] = canvas[1][:label_pos] + clip_label + canvas[1][label_pos+len(clip_label):]
                
                # Output the canvas
                for line in canvas:
                    result += line + "\n"
                
                # Add clip details
                if detail_level != 'minimal':
                    for i, obj in enumerate(track):
                        start_time = obj.start / fps
                        duration = obj.dur / fps
                        end_time = start_time + duration
                        
                        if detail_level == 'normal':
                            # Basic information
                            if isinstance(obj, TlVideo):
                                src_name = "Unknown"
                                if obj.src is not None and hasattr(obj.src, 'path'):
                                    src_name = obj.src.path.name
                                result += f"  {i+1}. Video: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | Source: {src_name}\n"
                            elif isinstance(obj, TlImage):
                                src_name = "Unknown"
                                if obj.src is not None and hasattr(obj.src, 'path'):
                                    src_name = obj.src.path.name
                                result += f"  {i+1}. Image: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | Source: {src_name}\n"
                            elif isinstance(obj, TlRect):
                                result += f"  {i+1}. Rectangle: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | Color: {obj.fill}\n"
                            elif isinstance(obj, TlText):
                                # Truncate text if too long
                                display_text = obj.text[:20] + "..." if len(obj.text) > 20 else obj.text
                                display_text = display_text.replace("\n", " ")
                                result += f"  {i+1}. Text: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | \"{display_text}\"\n"
                        elif detail_level == 'detailed':
                            # Detailed information
                            if isinstance(obj, TlVideo):
                                src_name = "Unknown"
                                if obj.src is not None and hasattr(obj.src, 'path'):
                                    src_name = obj.src.path.name
                                offset_time = obj.offset / fps
                                result += (f"  {i+1}. Video: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)\n"
                                          f"     Source: {src_name}, Offset: {offset_time:.2f}s, Speed: {obj.speed}x\n")
                            elif isinstance(obj, TlImage):
                                src_name = "Unknown"
                                if obj.src is not None and hasattr(obj.src, 'path'):
                                    src_name = obj.src.path.name
                                result += (f"  {i+1}. Image: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)\n"
                                          f"     Source: {src_name}, Position: ({obj.x}, {obj.y}), Width: {obj.width}, Opacity: {obj.opacity}\n")
                            elif isinstance(obj, TlRect):
                                result += (f"  {i+1}. Rectangle: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)\n"
                                          f"     Position: ({obj.x}, {obj.y}), Size: {obj.width}x{obj.height}, Color: {obj.fill}\n")
                            elif isinstance(obj, TlText):
                                # Truncate text if too long
                                display_text = obj.text[:40] + "..." if len(obj.text) > 40 else obj.text
                                display_text = display_text.replace("\n", " ")
                                result += (f"  {i+1}. Text: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)\n"
                                          f"     Content: \"{display_text}\", Position: ({obj.x}, {obj.y}), Font: {obj.font}/{obj.font_size}px, Style: {obj.style}\n")
                
                # Add a separator between tracks
                result += "\n"
        
        # Audio tracks
        if detail_level != 'minimal':
            result += "\nAudio Tracks:\n"
            if not timeline.a or all(not track for track in timeline.a):
                result += "  No audio clips\n"
            else:
                for track_idx, track in enumerate(timeline.a):
                    if not track:
                        continue
                        
                    # Track header
                    result += f"Track {track_idx+1}:\n"
                    
                    # Create timeline canvas for this track
                    canvas = [" " * width for _ in range(2)]  # 2 lines per track
                    
                    # Place clips on the canvas
                    for i, obj in enumerate(track):
                        # Calculate positions
                        start_pos = int((obj.start / total_frames) * (width - 1)) if total_frames > 0 else 0
                        end_pos = int(((obj.start + obj.dur) / total_frames) * (width - 1)) if total_frames > 0 else width
                        
                        # Ensure minimum width for visibility
                        if end_pos <= start_pos:
                            end_pos = start_pos + 1
                        end_pos = min(end_pos, width)
                        
                        # Draw clip on canvas
                        # Draw top line (clip boundary)
                        canvas[0] = canvas[0][:start_pos] + "+" + "~" * (end_pos - start_pos - 2) + "+" + canvas[0][end_pos:]
                        
                        # Draw bottom line (clip index)
                        clip_label = f"{i+1}"
                        label_pos = start_pos + (end_pos - start_pos) // 2 - len(clip_label) // 2
                        label_pos = max(start_pos, min(label_pos, end_pos - len(clip_label)))
                        
                        if end_pos - start_pos > len(clip_label):
                            canvas[1] = canvas[1][:label_pos] + clip_label + canvas[1][label_pos+len(clip_label):]
                    
                    # Output the canvas
                    for line in canvas:
                        result += line + "\n"
                    
                    # Add clip details
                    if detail_level == 'normal':
                        for i, obj in enumerate(track):
                            start_time = obj.start / fps
                            duration = obj.dur / fps
                            end_time = start_time + duration
                            src_name = "Unknown"
                            if obj.src is not None and hasattr(obj.src, 'path'):
                                src_name = obj.src.path.name
                            result += f"  {i+1}. Audio: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s) | Source: {src_name}\n"
                    elif detail_level == 'detailed':
                        for i, obj in enumerate(track):
                            start_time = obj.start / fps
                            duration = obj.dur / fps
                            end_time = start_time + duration
                            offset_time = obj.offset / fps
                            src_name = "Unknown"
                            if obj.src is not None and hasattr(obj.src, 'path'):
                                src_name = obj.src.path.name
                            result += (f"  {i+1}. Audio: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)\n"
                                     f"     Source: {src_name}, Offset: {offset_time:.2f}s, Speed: {obj.speed}x, Volume: {obj.volume}\n")
                    
                    # Add a separator between tracks
                    result += "\n"
        
        return result
        
    def convert_v1_to_v3(self, timeline_v1: v1) -> v3:
        """
        Convert a v1 timeline to a v3 timeline.
        
        Args:
            timeline_v1: v1 timeline object to convert
            
        Returns:
            v3 timeline object
        """
        try:
            # Get source information
            source = timeline_v1.source
            
            # Get framerate as Fraction
            if hasattr(source, 'video') and source.video:
                framerate = Fraction(source.video.fps).limit_denominator(1000)
                sample_rate = getattr(source.audio, 'samplerate', 48000) if hasattr(source, 'audio') else 48000
                width = source.video.width
                height = source.video.height
            else:
                framerate = Fraction(30, 1)
                sample_rate = 48000
                width = 1920
                height = 1080
                
            # Create empty v3 timeline with source settings
            timeline_v3 = v3(
                src=source,
                tb=framerate,
                sr=sample_rate,
                res=(width, height),
                background="#000000",
                v=[[]],
                a=[[]],
                v1=timeline_v1  # Store v1 reference
            )
            
            # Process the chunks to create video segments
            chunks = timeline_v1.chunks
            
            # Calculate total video duration in frames
            if hasattr(source, 'video') and source.video:
                total_frames = int(source.video.duration * source.video.fps)
            else:
                self._log_error("Source does not have valid video stream")
                return timeline_v3
            
            # Process chunks based on whether they are inverted
            is_inverted = getattr(chunks, 'inverted', False)
            
            if hasattr(chunks, 'chunks'):
                # Current position in output timeline (in frames)
                current_frame = 0
                
                # Convert chunks to video objects in v3 timeline
                for chunk in chunks.chunks:
                    start, end = chunk[:2]  # Some chunks might have a third value (speed)
                    
                    # Skip based on inversion mode
                    if is_inverted:
                        # If inverted, these are the sections to remove
                        continue
                        
                    # Calculate duration in frames
                    duration = end - start
                    
                    # Create video object for the chunk
                    video_obj = TlVideo(
                        start=current_frame,
                        dur=duration,
                        src=source,
                        offset=start,  # Start position in source
                        speed=1.0,     # Default speed
                        stream=0       # Main video stream
                    )
                    
                    # Add to the first video track
                    timeline_v3.v[0].append(video_obj)
                    
                    # Update current position
                    current_frame += duration
                    
                # Create audio objects for the same chunks if audio exists
                if hasattr(source, 'audio') and source.audio:
                    current_frame = 0
                    for chunk in chunks.chunks:
                        start, end = chunk[:2]
                        
                        # Skip based on inversion mode
                        if is_inverted:
                            continue
                            
                        # Calculate duration in frames
                        duration = end - start
                        
                        # Create audio object for the chunk
                        audio_obj = TlAudio(
                            start=current_frame,
                            dur=duration,
                            src=source,
                            offset=start,  # Start position in source
                            speed=1.0,     # Default speed
                            volume=1.0,    # Default volume
                            stream=0       # Main audio stream
                        )
                        
                        # Add to the first audio track
                        timeline_v3.a[0].append(audio_obj)
                        
                        # Update current position
                        current_frame += duration
            
            self._log_info(f"Successfully converted v1 timeline to v3 timeline")
            return timeline_v3
            
        except Exception as e:
            self._log_error(f"Error converting v1 to v3 timeline: {e}", exc_info=e)
            raise
            
    def convert_v3_to_v1(self, timeline_v3: v3) -> Optional[v1]:
        """
        Try to convert a v3 timeline to a v1 timeline.
        Only works for simple v3 timelines with a single source.
        
        Args:
            timeline_v3: v3 timeline object to convert
            
        Returns:
            v1 timeline object or None if conversion is not possible
        """
        try:
            # Check if timeline is already v1-compatible
            if timeline_v3.v1 is not None:
                self._log_info("Using existing v1 compatibility reference")
                return timeline_v3.v1
                
            # Check if all videos come from the same source
            sources = set()
            for track in timeline_v3.v:
                for obj in track:
                    if isinstance(obj, TlVideo):
                        sources.add(str(obj.src.path))
            
            if len(sources) != 1:
                self._log_error("Cannot convert to v1: Multiple different sources found")
                return None
                
            # Get the first video object to determine source
            source = None
            first_obj = None
            for track in timeline_v3.v:
                for obj in track:
                    if isinstance(obj, TlVideo):
                        source = obj.src
                        first_obj = obj
                        break
                if source:
                    break
                    
            if not source:
                self._log_error("Cannot convert to v1: No video objects found")
                return None
                
            # Create chunks from video objects
            chunk_list = []
            for track in timeline_v3.v:
                for obj in track:
                    if isinstance(obj, TlVideo):
                        # Convert to source frame coordinates
                        start = obj.offset
                        end = obj.offset + obj.dur
                        chunk_list.append((start, end))
            
            # Sort chunks by start time
            chunk_list.sort(key=lambda x: x[0])
            
            # Create v1 timeline
            chunks = UniformChunks(chunk_list, is_range=True)
            timeline_v1 = v1(source=source, chunks=chunks)
            
            self._log_info(f"Successfully converted v3 timeline to v1 timeline")
            return timeline_v1
            
        except Exception as e:
            self._log_error(f"Error converting v3 to v1 timeline: {e}", exc_info=e)
            return None
    
    def get_timeline_path(self, timeline_name: str, ensure_json_ext: bool = True) -> Path:
        """
        Get the proper path for a timeline file.
        
        Args:
            timeline_name: Base name for the timeline
            ensure_json_ext: Whether to ensure .json extension is present
            
        Returns:
            Path object for the timeline file
        """
        # Use the FileManager's timeline path method with channel context
        return file_mgr.get_timeline_path(timeline_name, self.channel_number)
    
    def load_timeline(self, timeline_name: str) -> Union[v1, v3, None]:
        """
        Load a timeline by name, handling path resolution.
        
        Args:
            timeline_name: Name of the timeline to load
            
        Returns:
            Timeline object or None if loading fails
        """
        timeline_path = self.get_timeline_path(timeline_name)
        return self.deserialize_timeline(timeline_path)
    
    def save_timeline(self, timeline: Union[v1, v3], timeline_name: str, 
                     description: str = "") -> bool:
        """
        Save a timeline by name, handling path resolution.
        
        Args:
            timeline: Timeline object to save
            timeline_name: Name to save the timeline as
            description: Optional description of the timeline
            
        Returns:
            True if saving was successful
        """
        try:
            timeline_path = self.get_timeline_path(timeline_name)
            self.serialize_timeline(timeline, timeline_path, description=description)
            return True
        except Exception as e:
            self._log_error(f"Error saving timeline {timeline_name}: {e}", exc_info=e)
            return False
    
    def add_captions_to_timeline(self, 
                             timeline: v3, 
                             captions_path: PathLike,
                             track_index: int = 1,
                             font: str = "Arial",
                             font_size: int = 36,
                             color: str = "#FFFFFF",
                             bg_color: str = "#00000080",
                             align: str = "center",
                             position: str = "bottom",
                             padding: int = 20,
                             max_caption_duration: float = 5.0) -> v3:
        """
        Add captions from an SRT file to a timeline as text elements.
        
        Args:
            timeline: v3 timeline object to add captions to
            captions_path: Path to the SRT captions file
            track_index: Video track index to add captions to (defaults to track 1)
            font: Font family to use for captions
            font_size: Font size in pixels
            color: Text color in hex format
            bg_color: Background color in hex format with alpha (80 = 50% opacity)
            align: Text alignment ("left", "center", "right")
            position: Vertical position ("top", "middle", "bottom")
            padding: Padding from the edge in pixels
            max_caption_duration: Maximum duration for a caption in seconds
            
        Returns:
            Updated timeline with caption text elements
        """
        try:
            # Ensure the timeline is v3
            if not isinstance(timeline, v3):
                self._log_error("Only v3 timelines support caption text elements")
                return timeline
                
            # Check if the captions file exists
            captions_path = file_mgr.normalize_path(captions_path)
            if not captions_path.exists():
                self._log_error(f"Captions file not found: {captions_path}")
                return timeline
                
            # Read the SRT file
            srt_content = file_mgr.read_text(captions_path)
            if not srt_content:
                self._log_error(f"Failed to read captions file: {captions_path}")
                return timeline
                
            # Parse the SRT content
            import re
            
            # SRT time format: 00:00:00,000 --> 00:00:00,000
            time_pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})'
            
            # Split the SRT file into caption blocks
            caption_blocks = re.split(r'\n\s*\n', srt_content.strip())
            
            # Get timeline properties
            fps = float(timeline.tb) if timeline.tb else 30.0
            width, height = timeline.res
            
            # Determine Y position based on requested position
            y_pos = 0
            if position == "bottom":
                y_pos = height - padding - font_size
            elif position == "middle":
                y_pos = height // 2
            elif position == "top":
                y_pos = padding
                
            # Ensure we have enough video tracks
            while len(timeline.v) <= track_index:
                timeline.v.append([])
                
            # Parse each caption block and add to timeline
            for block in caption_blocks:
                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue
                    
                # Find the timecode line using regex
                timecode_match = None
                for line in lines:
                    match = re.search(time_pattern, line)
                    if match:
                        timecode_match = match
                        break
                
                if not timecode_match:
                    continue
                    
                # Extract start and end times
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, timecode_match.groups())
                start_time = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                end_time = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                
                # Limit caption duration if needed
                if end_time - start_time > max_caption_duration:
                    end_time = start_time + max_caption_duration
                
                # Convert to frame numbers
                start_frame = int(start_time * fps)
                duration_frames = int((end_time - start_time) * fps)
                
                # Skip very short captions
                if duration_frames < 2:
                    continue
                
                # Extract the caption text (all lines after the timestamp)
                caption_text = ""
                capture_text = False
                for line in lines:
                    if capture_text:
                        caption_text += line + "\n"
                    elif re.search(time_pattern, line):
                        capture_text = True
                
                caption_text = caption_text.strip()
                if not caption_text:
                    continue
                
                # Create text element
                text_obj = TlText(
                    start=start_frame,
                    dur=duration_frames,
                    text=caption_text,
                    x="50%",  # Center horizontally
                    y=y_pos,
                    font=font,
                    font_size=font_size,
                    color=color,
                    bg_color=bg_color,
                    align=align,
                    opacity=1.0,
                    style="normal"
                )
                
                # Add to the specified video track
                timeline.v[track_index].append(text_obj)
            
            # Also add captions to metadata for reference
            if not hasattr(timeline, 'videoai_metadata'):
                timeline.videoai_metadata = {}
                
            if 'captions' not in timeline.videoai_metadata:
                timeline.videoai_metadata['captions'] = {
                    'path': str(captions_path),
                    'style': {
                        'font': font,
                        'font_size': font_size,
                        'color': color,
                        'bg_color': bg_color,
                        'align': align,
                        'position': position
                    }
                }
            
            self._log_info(f"Added {len(timeline.v[track_index])} caption elements to timeline")
            return timeline
            
        except Exception as e:
            self._log_error(f"Error adding captions to timeline: {e}", exc_info=e)
            return timeline
            
    def export_timeline_visualization(self, timeline: Union[v1, v3], 
                                    output_path: Optional[PathLike] = None,
                                    width: Optional[int] = None, 
                                    detail_level: Optional[str] = None) -> bool:
        """
        Export a timeline visualization to a text file.
        
        Args:
            timeline: Timeline object to visualize
            output_path: Path where to save the visualization (if None, uses timeline name)
            width: Width of the visualization in characters (if None, uses config value)
            detail_level: Level of detail ('minimal', 'normal', 'detailed')
                        (if None, uses config value)
            
        Returns:
            True if export was successful
        """
        try:
            # Get configuration
            timeline_config = self._get_timeline_config()
            viz_config = timeline_config.visualization
            
            # Use configuration defaults if not specified
            if width is None:
                width = viz_config.max_width
                
            if detail_level is None:
                detail_level = viz_config.default_detail_level
            
            # Generate the visualization
            visualization = self.visualize_timeline(timeline, width=width, detail_level=detail_level)
            
            # Determine the output path if not provided
            if output_path is None:
                # Try to get a name from the source file
                if isinstance(timeline, v1) and hasattr(timeline.source, 'path'):
                    name = timeline.source.path.stem
                else:
                    # Generate a timestamp-based name
                    from datetime import datetime
                    name = f"timeline_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Use the FileManager's timeline visualization path method
                output_path = file_mgr.get_timeline_visualization_path(
                    name, 
                    detail_level=detail_level,
                    channel_number=self.channel_number
                )
            else:
                output_path = file_mgr.normalize_path(output_path)
                file_mgr.ensure_dir_exists(output_path.parent)
            
            # Write the visualization to the file
            file_mgr.write_text(output_path, visualization)
            
            self._log_info(f"Timeline visualization exported to {output_path}")
            return True
            
        except Exception as e:
            self._log_error(f"Error exporting timeline visualization: {e}", exc_info=e)
            return False
    
    def print_timeline_summary(self, timeline: Union[v1, v3], 
                               detail_level: Optional[str] = 'minimal') -> None:
        """
        Print a summary of the timeline to the console.
        
        This is a convenience method that uses a minimal detail level visualization
        by default to provide a compact overview of the timeline.
        
        Args:
            timeline: Timeline object to summarize
            detail_level: Level of detail ('minimal', 'normal', 'detailed')
                        If None, uses 'minimal'
        """
        if detail_level is None:
            detail_level = 'minimal'
            
        # Get configuration for width
        timeline_config = self._get_timeline_config()
        width = timeline_config.visualization.max_width
        
        summary = self.visualize_timeline(timeline, width=width, detail_level=detail_level)
        print(summary)