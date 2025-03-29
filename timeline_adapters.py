"""
Timeline Adapters module for VideoAI project.

Provides adapter functions to convert between VideoAI timeline structures and
the formats required by auto-editor's export modules. This includes:

- Timeline structure conversion
- Format-specific adapters
- Metadata enhancement for export formats
- Video/audio track conversion
- Special handling for custom elements like TlText

Features:
- Convert VideoAI timelines to export-compatible formats 
- Metadata preservation between formats
- Format-specific validation and preparation
- Path normalization and resolution
- Custom element adaptation for professional formats

These adapters serve as the bridge between VideoAI's internal timeline
representations and the formats expected by professional video editing software.
"""
import os
import json
import traceback
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Set, Type, cast

# Import auto-editor components
from auto_editor.timeline import v1, v3, TlVideo, TlAudio, TlImage, TlRect
from auto_editor.ffwrapper import FileInfo, initFileInfo
from auto_editor.utils.log import Log

# Import VideoAI components
from file_manager import FileManager
import timeline_manager
from timeline_manager import TimelineManager, TlText

# Type definitions for better type hinting
PathLike = Union[str, Path]
V3Timeline = v3
V1Timeline = v1

# Initialize file manager
file_mgr = FileManager()

class TimelineAdapter:
    """
    Base adapter class for converting between VideoAI timelines and export formats.
    
    Provides common functionality for adapting timelines to various export formats.
    Subclasses implement format-specific adaptation logic.
    """
    
    def __init__(self, log: Optional[Log] = None):
        """
        Initialize the adapter.
        
        Args:
            log: Auto-editor Log object for output (creates one if None)
        """
        self.log = log or self._create_dummy_log()
        
        # Initialize default configuration
        self.config = {
            # Common settings
            "create_sidecar_files": True,
            "normalize_paths": True,
            "convert_custom_elements": True,
            
            # Path handling settings
            "relative_paths": True,
            "resolve_media_paths": True,
            "media_search_paths": [],
            
            # Format version settings
            "fcp7_version": 5,
            "fcp11_version": 5,
            "shotcut_version": "7.0",
            
            # Format-specific feature flags
            "fcp7_use_markers": True,
            "fcp11_use_roles": True,
            "fcp11_use_markers": True,
            "fcp11_use_compound_clips": True,
            "shotcut_use_filters": True
        }
        
        # Try to import logging system for VideoAI-specific logging
        try:
            from logging_system import Logger, LogLevel
            self.logger = Logger.get_logger("timeline_adapter")
            self.has_logging = True
        except ImportError:
            self.has_logging = False
            
    def set_config(self, config):
        """
        Configure the adapter with format-specific settings.
        
        Args:
            config: FormatSpecificExportConfig object or dictionary with configuration settings
        """
        # If config is a Pydantic model, convert to dictionary
        if hasattr(config, "model_dump"):
            # Use model_dump for Pydantic v2+
            config_dict = config.model_dump()
        elif hasattr(config, "dict"):
            # Fallback for Pydantic v1 (deprecated)
            config_dict = config.dict()
        else:
            config_dict = config
            
        # Update the configuration with provided values
        for key, value in config_dict.items():
            if key in self.config:
                self.config[key] = value
                
        self._log_info(f"Adapter configuration updated with {len(config_dict)} settings")
    
    def _create_dummy_log(self) -> Log:
        """Create a dummy logger instance"""
        return Log(is_debug=False, quiet=True)
    
    def _log_info(self, message: str) -> None:
        """Log an info message using the appropriate logging system."""
        if self.has_logging:
            self.logger.info(message)
        else:
            print(f"[TimelineAdapter] {message}")
            
    def _log_error(self, message: str, exc_info: Optional[Exception] = None) -> None:
        """Log an error message using the appropriate logging system."""
        if self.has_logging:
            self.logger.error(message, exc_info=exc_info is not None)
        else:
            print(f"[TimelineAdapter ERROR] {message}")
            if exc_info:
                traceback.print_exc()
    
    def _normalize_paths(self, timeline: Union[v1, v3]) -> Union[v1, v3]:
        """
        Normalize paths in a timeline to ensure compatibility with export formats.
        
        Args:
            timeline: Timeline object (v1 or v3)
            
        Returns:
            Timeline with normalized paths
        """
        # Skip normalization if disabled in configuration
        if not self.config["normalize_paths"]:
            self._log_info("Path normalization skipped (disabled in configuration)")
            return timeline
            
        use_relative_paths = self.config["relative_paths"]
        resolve_media_paths = self.config["resolve_media_paths"]
        search_paths = self.config["media_search_paths"]
        
        self._log_info(f"Normalizing paths (relative: {use_relative_paths}, resolve: {resolve_media_paths})")
        
        # Helper function to normalize a path
        def normalize_path(path_obj):
            if not path_obj:
                return path_obj
                
            # Handle FileInfo objects - extract the path
            if isinstance(path_obj, FileInfo):
                path = path_obj.path
            else:
                # Get a Path object that we can work with
                path = Path(path_obj)
            
            # Skip normalization for non-existent paths if not resolving media paths
            if not path.exists() and not resolve_media_paths:
                return path_obj
            
            # If using relative paths and path is absolute, try to make it relative to current directory
            if use_relative_paths and path.is_absolute():
                try:
                    # Try to make path relative to current directory
                    rel_path = path.relative_to(Path.cwd())
                    
                    # If it was a FileInfo object, create a new one with the updated path
                    if isinstance(path_obj, FileInfo):
                        # Create a copy of the FileInfo object with the updated path
                        # This is a simple approximation
                        new_info = FileInfo(
                            path=rel_path,
                            bitrate=path_obj.bitrate,
                            duration=path_obj.duration,
                            description=path_obj.description,
                            videos=path_obj.videos,
                            audios=path_obj.audios,
                            subtitles=path_obj.subtitles
                        )
                        return new_info
                    return rel_path
                except ValueError:
                    # Can't make it relative, use absolute path or resolve it
                    pass
                    
            # Resolve media path if enabled and path doesn't exist
            if resolve_media_paths and not path.exists():
                for search_path in search_paths:
                    # Try to find the file by name in search paths
                    test_path = Path(search_path) / path.name
                    if test_path.exists():
                        self._log_info(f"Resolved media path: {path} -> {test_path}")
                        
                        # If it was a FileInfo object, create a new one with the updated path
                        if isinstance(path_obj, FileInfo):
                            # Create a copy of the FileInfo object with the updated path
                            new_info = FileInfo(
                                path=test_path,
                                bitrate=path_obj.bitrate,
                                duration=path_obj.duration,
                                description=path_obj.description,
                                videos=path_obj.videos,
                                audios=path_obj.audios,
                                subtitles=path_obj.subtitles
                            )
                            return new_info
                        return test_path
            
            # Default: use the resolved absolute path
            resolved_path = path.resolve()
            if isinstance(path_obj, FileInfo):
                # Create a copy of the FileInfo object with the updated path
                new_info = FileInfo(
                    path=resolved_path,
                    bitrate=path_obj.bitrate,
                    duration=path_obj.duration,
                    description=path_obj.description,
                    videos=path_obj.videos,
                    audios=path_obj.audios,
                    subtitles=path_obj.subtitles
                )
                return new_info
            return resolved_path
        
        # For v1 timeline, normalize the source path
        if isinstance(timeline, v1):
            if hasattr(timeline.source, 'path') and timeline.source.path:
                # Create a copy to avoid modifying the original
                import copy
                timeline_copy = copy.deepcopy(timeline)
                timeline_copy.source.path = normalize_path(timeline_copy.source.path)
                return timeline_copy
        
        # For v3 timeline, normalize paths in all tracks
        elif isinstance(timeline, v3):
            # Create a copy to avoid modifying the original
            import copy
            timeline_copy = copy.deepcopy(timeline)
            
            # Normalize source path if available
            if hasattr(timeline_copy, 'src') and timeline_copy.src:
                if isinstance(timeline_copy.src, FileInfo):
                    timeline_copy.src = normalize_path(timeline_copy.src)
                elif hasattr(timeline_copy.src, 'path'):
                    timeline_copy.src.path = normalize_path(timeline_copy.src.path)
            
            # Process video tracks
            for track_idx, track in enumerate(timeline_copy.v):
                for clip_idx, clip in enumerate(track):
                    if hasattr(clip, 'src') and clip.src:
                        if isinstance(clip.src, FileInfo):
                            timeline_copy.v[track_idx][clip_idx].src = normalize_path(clip.src)
                        elif hasattr(clip.src, 'path'):
                            timeline_copy.v[track_idx][clip_idx].src.path = normalize_path(clip.src.path)
            
            # Process audio tracks
            for track_idx, track in enumerate(timeline_copy.a):
                for clip_idx, clip in enumerate(track):
                    if hasattr(clip, 'src') and clip.src:
                        if isinstance(clip.src, FileInfo):
                            timeline_copy.a[track_idx][clip_idx].src = normalize_path(clip.src)
                        elif hasattr(clip.src, 'path'):
                            timeline_copy.a[track_idx][clip_idx].src.path = normalize_path(clip.src.path)
            
            return timeline_copy
        
        # If no normalization was needed or possible, return the original
        return timeline

    def preserve_metadata(self, timeline: Union[v1, v3], format_type: str) -> Dict[str, Any]:
        """
        Extract and format VideoAI metadata for preservation in export formats.
        
        Args:
            timeline: The timeline containing metadata
            format_type: Export format type ("json", "fcp7", "fcp11", "shotcut")
            
        Returns:
            Dictionary with formatted metadata
        """
        # Start with a base metadata structure
        metadata = {
            "videoai_export": {
                "format": format_type,
                "version": "1.0", 
                "timestamp": self._get_timestamp(),
                "original_metadata": {},
                "config_used": {}
            }
        }
        
        # Include the configuration used to generate this export
        # Filter only the keys relevant to the current format
        format_prefix = f"{format_type}_"
        format_config = {}
        
        for key, value in self.config.items():
            # Include common config options and format-specific ones
            if key in ["create_sidecar_files", "normalize_paths", "convert_custom_elements", 
                       "relative_paths", "resolve_media_paths"] or key.startswith(format_prefix):
                format_config[key] = value
                
        metadata["videoai_export"]["config_used"] = format_config
        
        # Extract original metadata if present
        if isinstance(timeline, v1) and hasattr(timeline, 'videoai_metadata'):
            metadata["videoai_export"]["original_metadata"] = getattr(timeline, 'videoai_metadata', {})
        elif isinstance(timeline, v3):
            if hasattr(timeline, 'v1') and timeline.v1 and hasattr(timeline.v1, 'videoai_metadata'):
                metadata["videoai_export"]["original_metadata"] = getattr(timeline.v1, 'videoai_metadata', {})
            elif hasattr(timeline, 'videoai_metadata'):
                metadata["videoai_export"]["original_metadata"] = getattr(timeline, 'videoai_metadata', {})
        
        return metadata
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def find_media_files(self, timeline: Union[v1, v3]) -> Set[Path]:
        """
        Find all media files used in a timeline.
        
        Args:
            timeline: Timeline object (v1 or v3)
            
        Returns:
            Set of Path objects for all media files
        """
        media_files = set()
        
        # For v1 timeline, add the source file
        if isinstance(timeline, v1):
            if hasattr(timeline.source, 'path') and timeline.source.path:
                media_files.add(Path(timeline.source.path).resolve())
        
        # For v3 timeline, add all sources from tracks
        elif isinstance(timeline, v3):
            # Add main source if available
            if hasattr(timeline, 'src') and timeline.src:
                if isinstance(timeline.src, FileInfo):
                    media_files.add(timeline.src.path.resolve())
                elif hasattr(timeline.src, 'path'):
                    media_files.add(Path(timeline.src.path).resolve())
            
            # Process video tracks
            for track in timeline.v:
                for clip in track:
                    if hasattr(clip, 'src') and clip.src:
                        if isinstance(clip.src, FileInfo):
                            media_files.add(clip.src.path.resolve())
                        elif hasattr(clip.src, 'path'):
                            media_files.add(Path(clip.src.path).resolve())
            
            # Process audio tracks
            for track in timeline.a:
                for clip in track:
                    if hasattr(clip, 'src') and clip.src:
                        if isinstance(clip.src, FileInfo):
                            media_files.add(clip.src.path.resolve())
                        elif hasattr(clip.src, 'path'):
                            media_files.add(Path(clip.src.path).resolve())
        
        return media_files
    
    def adapt_timeline(self, timeline: Union[v1, v3], format_type: str) -> Tuple[Union[v1, v3], Dict[str, Any]]:
        """
        Adapt a timeline for a specific export format.
        
        Args:
            timeline: Timeline object (v1 or v3)
            format_type: Export format type ("json", "fcp7", "fcp11", "shotcut")
            
        Returns:
            Tuple of (adapted timeline, format-specific metadata)
        """
        self._log_info(f"Adapting timeline for {format_type} format")
        
        # Store the format type for reference in other methods
        self.format_type = format_type
        
        # Copy the timeline to avoid modifying the original
        import copy
        timeline_copy = copy.deepcopy(timeline)
        
        # Normalize paths
        timeline_copy = self._normalize_paths(timeline_copy)
        
        # Preserve metadata
        metadata = self.preserve_metadata(timeline_copy, format_type)
        
        # Additional processing based on configuration
        self._log_info(f"Using configuration: create_sidecar_files={self.config['create_sidecar_files']}, convert_custom_elements={self.config['convert_custom_elements']}")
        
        return timeline_copy, metadata
        
    def create_metadata_sidecar(self, timeline: Union[v1, v3], metadata: Dict[str, Any], 
                                output_path: PathLike) -> Optional[Path]:
        """
        Create a sidecar file to preserve VideoAI metadata in formats that don't support embedded metadata.
        
        Args:
            timeline: The timeline that was exported
            metadata: Dictionary with metadata to preserve
            output_path: Path to the exported file
            
        Returns:
            Path to the sidecar file or None if sidecar files are disabled
        """
        # Skip sidecar file creation if disabled in configuration
        if not self.config["create_sidecar_files"]:
            self._log_info("Sidecar file creation skipped (disabled in configuration)")
            return None
            
        try:
            # Convert output path to Path object
            output_file = Path(output_path)
            
            # Create a sidecar file path with .videoai.json extension
            sidecar_path = output_file.with_suffix(f"{output_file.suffix}.videoai.json")
            
            # Add timestamp and version to metadata
            metadata["videoai_export"]["sidecar_created"] = self._get_timestamp()
            metadata["videoai_export"]["sidecar_version"] = "1.0"
            
            # Write metadata to sidecar file
            with open(sidecar_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self._log_info(f"Created metadata sidecar file: {sidecar_path}")
            return sidecar_path
            
        except Exception as e:
            self._log_error(f"Error creating metadata sidecar file: {e}")
            return None


class JSONAdapter(TimelineAdapter):
    """
    Adapter for JSON format export.
    
    The JSON format is the most flexible and can represent both v1 and v3 timelines
    with all VideoAI custom elements.
    """
    
    def adapt_timeline(self, timeline: Union[v1, v3], format_type: str = "json") -> Tuple[Union[v1, v3], Dict[str, Any]]:
        """
        Adapt a timeline for JSON export format.
        
        Args:
            timeline: Timeline object (v1 or v3)
            format_type: Export format type (always "json" for this adapter)
            
        Returns:
            Tuple of (adapted timeline, format-specific metadata)
        """
        # Call the base method for basic adaptation
        timeline, metadata = super().adapt_timeline(timeline, format_type)
        
        # JSON format supports all timeline elements, so no additional adaptation is needed
        # Just ensure paths are normalized
        
        # Add JSON-specific metadata
        metadata["videoai_export"]["format_details"] = {
            "supports_custom_elements": True,
            "preserves_all_metadata": True
        }
        
        # Add timeline type information
        if isinstance(timeline, v1):
            metadata["videoai_export"]["timeline_type"] = "v1"
        else:
            metadata["videoai_export"]["timeline_type"] = "v3"
        
        return timeline, metadata
    
    def inject_metadata(self, timeline_dict: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject metadata into a timeline dictionary for JSON export.
        
        Args:
            timeline_dict: Dictionary representation of a timeline
            metadata: Metadata to inject
            
        Returns:
            Updated timeline dictionary with injected metadata
        """
        # For JSON format, we can directly add our metadata to the dictionary
        timeline_dict["videoai_metadata"] = metadata
        return timeline_dict


class ProfessionalFormatAdapter(TimelineAdapter):
    """
    Base adapter for professional format exports (FCP7, FCP11, Shotcut).
    
    Professional formats have more restrictions and require special handling for
    custom elements and metadata.
    """
    
    def adapt_timeline(self, timeline: Union[v1, v3], format_type: str) -> Tuple[v3, Dict[str, Any]]:
        """
        Adapt a timeline for professional export formats.
        
        Args:
            timeline: Timeline object (v1 or v3)
            format_type: Export format type ("fcp7", "fcp11", "shotcut")
            
        Returns:
            Tuple of (adapted timeline, format-specific metadata)
        """
        # Call the base method for basic adaptation
        timeline, metadata = super().adapt_timeline(timeline, format_type)
        
        # Professional formats only support v3 timelines
        if isinstance(timeline, v1):
            raise ValueError(f"Format {format_type} requires a v3 timeline, got v1")
        
        # We know it's a v3 timeline at this point
        v3_timeline = cast(v3, timeline)
        
        # Convert TlText elements to compatible alternatives (images or rects)
        v3_timeline = self._convert_custom_elements(v3_timeline)
        
        # Add format-specific metadata
        metadata["videoai_export"]["format_details"] = {
            "supports_custom_elements": False,
            "custom_elements_converted": True,
            "format": format_type
        }
        
        return v3_timeline, metadata
    
    def _convert_custom_elements(self, timeline: v3) -> v3:
        """
        Convert custom elements (like TlText) to compatible alternatives.
        
        Args:
            timeline: v3 timeline with possible custom elements
            
        Returns:
            v3 timeline with custom elements converted to standard elements
        """
        # Create a copy to avoid modifying the original
        import copy
        timeline_copy = copy.deepcopy(timeline)
        
        # Process video tracks
        has_text_conversions = False
        for track_idx, track in enumerate(timeline_copy.v):
            new_track = []
            for clip in track:
                if isinstance(clip, TlText):
                    # Convert TlText to TlRect as a placeholder
                    rect = self._text_to_rect(clip)
                    new_track.append(rect)
                    has_text_conversions = True
                else:
                    new_track.append(clip)
            timeline_copy.v[track_idx] = new_track
        
        # Log a warning if we converted text elements
        if has_text_conversions:
            self._log_info("Converted TlText elements to compatible alternatives for professional format export")
        
        return timeline_copy
    
    def _text_to_rect(self, text: TlText) -> TlRect:
        """
        Convert a TlText element to a TlRect placeholder.
        
        Used for formats that don't support native text elements.
        For FCP11 format, TlText objects are now handled directly.
        
        Args:
            text: TlText element to convert
            
        Returns:
            TlRect placeholder representing the text area
        """
        # Create a placeholder rectangle with the same timing as the text
        return TlRect(
            start=text.start,
            dur=text.dur,
            x=text.x,
            y=text.y,
            width=200,  # Rough estimate of text width
            height=50,  # Rough estimate of text height
            fill="#808080"  # Gray placeholder color
        )
    
    def create_metadata_sidecar(self, timeline: v3, metadata: Dict[str, Any], output_path: Path) -> Optional[Path]:
        """
        Create a sidecar file with metadata for formats that don't support embedded metadata.
        
        Args:
            timeline: Timeline that was exported
            metadata: Metadata to preserve
            output_path: Path to the exported file
            
        Returns:
            Path to the sidecar file or None if creation failed
        """
        try:
            # Create a JSON sidecar with the same base name + .videoai.json extension
            # This needs to match the exact path that the test is looking for
            extension = output_path.suffix
            sidecar_path = output_path.with_suffix(f"{extension}.videoai.json")
            
            # Prepare sidecar content with detailed metadata
            sidecar_content = {
                "videoai_sidecar": {
                    "version": "1.0",
                    "timestamp": self._get_timestamp(),
                    "export_path": str(output_path),
                    "metadata": metadata
                }
            }
            
            # Add timeline details
            sidecar_content["videoai_sidecar"]["timeline_details"] = {
                "resolution": list(timeline.res),
                "timebase": f"{timeline.tb.numerator}/{timeline.tb.denominator}",
                "tracks": {
                    "video": len(timeline.v),
                    "audio": len(timeline.a)
                }
            }
            
            # Write the sidecar file
            with open(sidecar_path, 'w') as f:
                json.dump(sidecar_content, f, indent=2)
            
            return sidecar_path
            
        except Exception as e:
            self._log_error(f"Error creating metadata sidecar: {e}", exc_info=e)
            return None


class FCP7Adapter(ProfessionalFormatAdapter):
    """
    Adapter for Final Cut Pro 7 XML format export.
    
    FCP7 format is also used by Adobe Premiere Pro and has specific requirements
    for timeline structures.
    """
    
    def adapt_timeline(self, timeline: Union[v1, v3], format_type: str = "fcp7") -> Tuple[v3, Dict[str, Any]]:
        """
        Adapt a timeline for FCP7 XML export format.
        
        Args:
            timeline: Timeline object (v1 or v3)
            format_type: Export format type (always "fcp7" for this adapter)
            
        Returns:
            Tuple of (adapted timeline, format-specific metadata)
        """
        # Call the base method for professional format adaptation
        timeline, metadata = super().adapt_timeline(timeline, format_type)
        
        # We know it's a v3 timeline at this point
        v3_timeline = cast(v3, timeline)
        
        # Add FCP7-specific metadata
        metadata["videoai_export"]["format_details"].update({
            "premiere_compatible": True,
            "supports_speed_effects": True
        })
        
        return v3_timeline, metadata


class FCP11Adapter(ProfessionalFormatAdapter):
    """
    Adapter for Final Cut Pro X XML format export.
    
    FCP11 (FCPXML) format is used by Final Cut Pro X and has more advanced
    timeline capabilities than FCP7, including support for text elements.
    """
    
    def adapt_timeline(self, timeline: Union[v1, v3], format_type: str = "fcp11") -> Tuple[v3, Dict[str, Any]]:
        """
        Adapt a timeline for FCP11 XML export format.
        
        Args:
            timeline: Timeline object (v1 or v3)
            format_type: Export format type (always "fcp11" for this adapter)
            
        Returns:
            Tuple of (adapted timeline, format-specific metadata)
        """
        # Call the base class's adapt_timeline but skip the parent class's implementation
        # (we want to avoid converting TlText objects for FCP11 format)
        timeline, metadata = TimelineAdapter.adapt_timeline(self, timeline, format_type)
        
        # Professional formats only support v3 timelines
        if isinstance(timeline, v1):
            raise ValueError(f"Format {format_type} requires a v3 timeline, got v1")
        
        # We know it's a v3 timeline at this point
        v3_timeline = cast(v3, timeline)
        
        # Add FCP11-specific metadata
        metadata["videoai_export"]["format_details"] = {
            "supports_custom_elements": True,
            "supports_text_elements": True,
            "custom_elements_converted": False,
            "format": format_type,
            "fcpx_version": 5,  # FCP 10.5 and newer
            "supports_roles": True,
            "supports_markers": True
        }
        
        # Check if timeline has any TlText objects
        has_text_objects = False
        for track in v3_timeline.v:
            for clip in track:
                if hasattr(timeline_manager, 'TlText') and isinstance(clip, timeline_manager.TlText):
                    has_text_objects = True
                    break
            if has_text_objects:
                break
                
        # If we have text objects, add this information to the metadata
        if has_text_objects:
            metadata["videoai_export"]["has_text_objects"] = True
        
        return v3_timeline, metadata


class ShotcutAdapter(ProfessionalFormatAdapter):
    """
    Adapter for Shotcut MLT format export.
    
    Shotcut MLT format is used by the Shotcut video editor and has different
    requirements than FCP formats.
    """
    
    def adapt_timeline(self, timeline: Union[v1, v3], format_type: str = "shotcut") -> Tuple[v3, Dict[str, Any]]:
        """
        Adapt a timeline for Shotcut MLT export format.
        
        Args:
            timeline: Timeline object (v1 or v3)
            format_type: Export format type (always "shotcut" for this adapter)
            
        Returns:
            Tuple of (adapted timeline, format-specific metadata)
        """
        # Call the base method for professional format adaptation
        timeline, metadata = super().adapt_timeline(timeline, format_type)
        
        # We know it's a v3 timeline at this point
        v3_timeline = cast(v3, timeline)
        
        # Add Shotcut-specific metadata
        metadata["videoai_export"]["format_details"].update({
            "mlt_version": "7.0",
            "supports_filters": True
        })
        
        return v3_timeline, metadata


class AdapterFactory:
    """
    Factory for creating the appropriate adapter for a given export format.
    """
    
    @staticmethod
    def create_adapter(format_type: str, log: Optional[Log] = None) -> TimelineAdapter:
        """
        Create the appropriate adapter for the specified format.
        
        Args:
            format_type: Format type ("json", "fcp7", "fcp11", "shotcut")
            log: Optional Log object
            
        Returns:
            TimelineAdapter instance for the specified format
        """
        format_type = format_type.lower()
        
        if format_type == "json":
            return JSONAdapter(log)
        elif format_type == "fcp7":
            return FCP7Adapter(log)
        elif format_type == "fcp11":
            return FCP11Adapter(log)
        elif format_type == "shotcut":
            return ShotcutAdapter(log)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")


# Additional utility functions for format-specific adjustments

def get_adapter_capabilities() -> Dict[str, Dict[str, Any]]:
    """
    Get a dictionary of adapter capabilities for each supported format.
    
    Returns:
        Dictionary mapping format names to their capabilities
    """
    return {
        "json": {
            "supports_v1": True,
            "supports_v3": True,
            "supports_custom_elements": True,
            "supports_metadata": True,
            "description": "Auto-Editor JSON format with full VideoAI capabilities"
        },
        "fcp7": {
            "supports_v1": False,
            "supports_v3": True,
            "supports_custom_elements": False,
            "supports_metadata": False,
            "description": "Final Cut Pro 7 XML format (also used by Adobe Premiere Pro)",
            "special_features": ["speed_effects", "transitions"]
        },
        "fcp11": {
            "supports_v1": False,
            "supports_v3": True,
            "supports_custom_elements": True,
            "supports_text_elements": True,
            "supports_metadata": False,
            "description": "Final Cut Pro X XML format",
            "special_features": ["roles", "markers", "compound_clips", "titles"]
        },
        "shotcut": {
            "supports_v1": False,
            "supports_v3": True,
            "supports_custom_elements": False,
            "supports_metadata": False,
            "description": "Shotcut MLT format",
            "special_features": ["filters", "transitions"]
        }
    }