"""
VideoAI Configuration Module

This module defines the configuration settings for the VideoAI project using Pydantic models
to ensure type safety and validation. It loads settings from environment variables and
provides default values for all settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from file_manager import FileManager

# Load environment variables
load_dotenv(override=True)

# Initialize the file manager
file_mgr = FileManager()

# with open('topics_covered.json', 'r') as f:
#     topics_covered = json.load(f)
# Base project directory (use file manager's base_dir)
BASE_DIR = file_mgr.base_dir

class OpenAISettings(BaseModel):
    """OpenAI API settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    script_model: str = Field(default="gpt-4o")
    title_desc_model: str = Field(default="gpt-4o")
    video_edit_model: str = Field(default="gpt-4.1-mini")
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=4000)

class ElevenLabsSettings(BaseModel):
    """ElevenLabs API settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    model_id: str = Field(default="eleven_multilingual_v2")
    default_voice_id: str = Field(default="UgBBYS2sOqTuMpoF3BR0")
    stability: float = Field(default=0.35)
    similarity_boost: float = Field(default=0.55)
    style: float = Field(default=0.10)
    use_speaker_boost: bool = Field(default=True)

class YoutubeSettings(BaseModel):
    """YouTube API settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    client_id: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_ID", ""))
    client_secret: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET", ""))
    privacy_status: str = Field(default="private")
    category_id: str = Field(default="28")  # Science & Technology

class ChannelExportConfig(BaseModel):
    """Channel-specific export configuration overrides"""
    export_directory: Optional[str] = Field(default=None)
    default_format: Optional[str] = Field(default=None)
    default_preset: Optional[str] = Field(default=None)
    auto_analyze: Optional[bool] = Field(default=None)
    validate_before_export: Optional[bool] = Field(default=None)
    on_incompatible_elements: Optional[str] = Field(default=None)
    
    # Format-specific overrides
    create_sidecar_files: Optional[bool] = Field(default=None)
    normalize_paths: Optional[bool] = Field(default=None)
    convert_custom_elements: Optional[bool] = Field(default=None)
    relative_paths: Optional[bool] = Field(default=None)
    resolve_media_paths: Optional[bool] = Field(default=None)
    media_search_paths: Optional[List[str]] = Field(default=None)

class ChannelTimelineConfig(BaseModel):
    """Channel-specific timeline configuration overrides"""
    storage_directory: Optional[str] = Field(default=None)
    visualization_directory: Optional[str] = Field(default=None)
    default_detail_level: Optional[str] = Field(default=None)
    default_framerate: Optional[int] = Field(default=None)
    default_width: Optional[int] = Field(default=None)
    default_height: Optional[int] = Field(default=None)
    default_background: Optional[str] = Field(default=None)
    
    # Export overrides
    export: Optional[ChannelExportConfig] = Field(default=None)
    
    # Rendering overrides
    rendering_enabled: Optional[bool] = Field(default=None)
    prefer_direct_rendering: Optional[bool] = Field(default=None)
    compatibility_mode: Optional[bool] = Field(default=None)
    force_fallback: Optional[bool] = Field(default=None)
    
    # Performance monitoring overrides
    enable_performance_monitoring: Optional[bool] = Field(default=None)
    track_memory_usage: Optional[bool] = Field(default=None)
    save_performance_reports: Optional[bool] = Field(default=None)
    performance_output_dir: Optional[str] = Field(default=None)

class ChannelConfig(BaseModel):
    """Channel-specific configuration"""
    name: str
    voice_id: str
    youtube_credentials_file: str
    youtube_info_file: str
    privacy_status: str = Field(default="private")
    # Channel-specific timeline configuration (overrides global settings)
    timeline: Optional[ChannelTimelineConfig] = Field(default=None)

class FilePathConfig(BaseModel):
    """File path configuration"""
    script_file: str = Field(default=f"generated_script.txt")
    voice_file: str = Field(default=f"voice/generated_voice.mp3")
    captions_file: str = Field(default=f"generated_voice.srt")
    output_video_file: str = Field(default=f"output_video.mp4")
    final_video_file: str = Field(default=f"final_output.mp4")
    final_subtitled_video_file: str = Field(default=f"final_output_with_subtitles.mp4")
    clips_metadata_file: str = Field(default="video/video-catalog-adjusted-2500.csv")
    clips_directory: str = Field(default="clips")
    background_music_directory: str = Field(default="background_music")
    output_directory: str = Field(default="outputs")
    
class VideoEditConfig(BaseModel):
    """Video editing configuration"""
    max_clip_duration: int = Field(default=10)
    default_clip_duration: int = Field(default=10)
    voice_volume: float = Field(default=1.4)
    background_music_volume: float = Field(default=0.4)  # Increased for better audibility
    subtitle_font: str = Field(default="DIN Condensed Bold")
    subtitle_font_size: int = Field(default=13)

class ScriptGenerationConfig(BaseModel):
    """Script generation configuration"""
    min_words: int = Field(default=50)
    max_words: int = Field(default=100)
    target_audience: str = Field(default="fact enthusiasts")
    tone: str = Field(default="informative yet conversational")
    style: str = Field(default="clear, engaging, and accessible")

class TimelineVisualizationConfig(BaseModel):
    """Timeline visualization configuration"""
    default_detail_level: str = Field(default="normal")
    max_width: int = Field(default=100)
    time_markers: int = Field(default=5)
    export_format: str = Field(default="txt")
    
class TimelineSerializationConfig(BaseModel):
    """Timeline serialization configuration"""
    format: str = Field(default="json")
    compression: bool = Field(default=False)
    include_metadata: bool = Field(default=True)
    validate_schema: bool = Field(default=True)
    auto_backup: bool = Field(default=True)

class FormatSpecificExportConfig(BaseModel):
    """Format-specific export configuration"""
    # Common settings
    create_sidecar_files: bool = Field(default=True, description="Create sidecar files for preserving VideoAI metadata")
    normalize_paths: bool = Field(default=True, description="Normalize media file paths for cross-platform compatibility")
    convert_custom_elements: bool = Field(default=True, description="Convert custom VideoAI elements to format-compatible elements")
    
    # Format version settings
    fcp7_version: int = Field(default=5, description="FCP7 XML schema version")
    fcp11_version: int = Field(default=5, description="FCP11 XML schema version (5+ for FCPX)") 
    shotcut_version: str = Field(default="7.0", description="Shotcut MLT version")
    
    # Format-specific feature flags
    fcp7_use_markers: bool = Field(default=True, description="Include markers in FCP7 exports")
    fcp11_use_roles: bool = Field(default=True, description="Use roles in FCP11 exports")
    fcp11_use_markers: bool = Field(default=True, description="Include markers in FCP11 exports")
    fcp11_use_compound_clips: bool = Field(default=True, description="Use compound clips in FCP11 exports")
    shotcut_use_filters: bool = Field(default=True, description="Include filters in Shotcut exports")
    
    # Path handling settings
    relative_paths: bool = Field(default=True, description="Use relative paths when possible")
    resolve_media_paths: bool = Field(default=True, description="Attempt to resolve missing media files")
    media_search_paths: List[str] = Field(default_factory=list, description="Additional paths to search for media files")
    
    # Compatibility settings
    ensure_dtd_compatibility: bool = Field(default=True, description="Ensure compatibility with format DTD validation by using older, more compatible schema versions")

class TimelineRenderingConfig(BaseModel):
    """Timeline rendering configuration"""
    # Feature flags for rendering capabilities
    enabled: bool = Field(default=False)  # Master switch for timeline-based rendering
    prefer_direct_rendering: bool = Field(default=True)  # Prefer direct rendering over fallback when possible
    
    # Rendering settings
    max_memory_mb: int = Field(default=4000)  # Maximum memory usage for rendering
    parallel_processing: bool = Field(default=True)  # Enable parallel processing for rendering
    error_recovery: bool = Field(default=True)  # Attempt to recover from rendering errors
    
    # Compatibility settings
    compatibility_mode: bool = Field(default=True)  # Enable backward compatibility mode
    force_fallback: bool = Field(default=False)  # Force using fallback rendering
    
    # Performance monitoring settings
    enable_performance_monitoring: bool = Field(default=True)  # Enable performance monitoring
    track_memory_usage: bool = Field(default=True)  # Track memory usage during rendering
    save_performance_reports: bool = Field(default=True)  # Save performance reports to disk
    performance_output_dir: str = Field(default="outputs/performance")  # Directory for performance reports
    
class ExportPresets(BaseModel):
    """Predefined export presets for common scenarios"""
    # Default preset (balanced settings)
    default: Dict[str, Any] = Field(default_factory=lambda: {
        "create_sidecar_files": True,
        "normalize_paths": True,
        "convert_custom_elements": True,
        "relative_paths": True
    })
    
    # Compatibility preset (maximum compatibility with older software)
    compatibility: Dict[str, Any] = Field(default_factory=lambda: {
        "create_sidecar_files": True,
        "normalize_paths": True,
        "convert_custom_elements": True,
        "fcp7_version": 4,
        "fcp11_version": 4,
        "shotcut_version": "6.0",
        "fcp11_use_compound_clips": False,
        "relative_paths": True,
        "resolve_media_paths": True
    })
    
    # Professional preset (best quality, modern features)
    professional: Dict[str, Any] = Field(default_factory=lambda: {
        "create_sidecar_files": True,
        "normalize_paths": True,
        "convert_custom_elements": True,
        "fcp7_version": 5,
        "fcp11_version": 6,
        "fcp11_use_roles": True,
        "fcp11_use_markers": True,
        "fcp11_use_compound_clips": True,
        "shotcut_use_filters": True,
        "relative_paths": False,
        "resolve_media_paths": True
    })
    
    # Minimal preset (basic settings, no extra features)
    minimal: Dict[str, Any] = Field(default_factory=lambda: {
        "create_sidecar_files": False,
        "normalize_paths": True,
        "convert_custom_elements": True,
        "fcp7_use_markers": False,
        "fcp11_use_roles": False,
        "fcp11_use_markers": False,
        "fcp11_use_compound_clips": False,
        "shotcut_use_filters": False,
        "relative_paths": True,
        "resolve_media_paths": False
    })

class ExportConfig(BaseModel):
    """Export configuration for timeline export"""
    # Export directory for exported files
    export_directory: str = Field(default="exports")
    
    # Default export format
    default_format: str = Field(default="json", description="Default export format (json, fcp7, fcp11, shotcut)")
    
    # Default export preset
    default_preset: str = Field(default="default", description="Default export preset (default, compatibility, professional, minimal)")
    
    # Format-specific configuration
    format_config: FormatSpecificExportConfig = Field(default_factory=FormatSpecificExportConfig)
    
    # Predefined export presets
    presets: ExportPresets = Field(default_factory=ExportPresets)
    
    # Auto-analyze timeline before export
    auto_analyze: bool = Field(default=True, description="Analyze timeline for export compatibility before exporting")
    
    # Default behavior for handling incompatible elements
    on_incompatible_elements: str = Field(
        default="convert",
        description="What to do with incompatible elements: 'convert', 'remove', 'error'"
    )
    
    # Validation rules
    validate_before_export: bool = Field(default=True, description="Validate timeline before export")
    max_validation_errors: int = Field(default=10, description="Maximum number of validation errors to report")
    stop_on_validation_error: bool = Field(default=False, description="Stop export on validation error")

class TimelineConfig(BaseModel):
    """Timeline configuration for creation, management and serialization"""
    # Timeline storage location
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
    
    # Export settings
    export: ExportConfig = Field(default_factory=ExportConfig)
    
    # Rendering settings
    rendering: TimelineRenderingConfig = Field(default_factory=TimelineRenderingConfig)
    
    # Advanced settings
    auto_save_interval: int = Field(default=300)  # in seconds, 0 to disable
    max_undo_steps: int = Field(default=10)  # 0 to disable undo history

class AppConfig(BaseSettings):
    """Main application configuration"""
    # API settings
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    elevenlabs: ElevenLabsSettings = Field(default_factory=ElevenLabsSettings)
    youtube: YoutubeSettings = Field(default_factory=YoutubeSettings)
    
    # File paths
    file_paths: FilePathConfig = Field(default_factory=FilePathConfig)
    
    # Video editing settings
    video_edit: VideoEditConfig = Field(default_factory=VideoEditConfig)
    
    # Script generation settings
    script_generation: ScriptGenerationConfig = Field(default_factory=ScriptGenerationConfig)
    
    # Timeline settings
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    
    # Channel configurations
    channels: Dict[int, ChannelConfig] = Field(default_factory=lambda: {
        1: ChannelConfig(
            name="Channel 1",
            voice_id="UgBBYS2sOqTuMpoF3BR0",
            #"29vD33N1CtxCmqQRPOHJ",
            youtube_credentials_file="youtube_token_channel1.json",
            youtube_info_file="youtube_info_channel1.json",
            # Channel-specific timeline configuration overrides
            timeline=ChannelTimelineConfig(
                default_detail_level="detailed",
                default_width=900,
                default_height=1600,
                default_framerate=30,
                # Export configuration overrides
                export=ChannelExportConfig(
                    default_format="fcp11",  # Use FCP11 as default format for this channel
                    default_preset="professional",  # Use professional preset
                    export_directory="exports/channel1",  # Custom export directory
                    create_sidecar_files=True,  # Always create sidecar files
                    media_search_paths=[  # Channel-specific media search paths
                        "outputs/channel_1/media",
                        "clips/filtered_videos"
                    ]
                )
            )
        ),
        2: ChannelConfig(
            name="Channel 2",
            voice_id="EXAVITQu4vr4xnSDxMaL",
            youtube_credentials_file="youtube_token_channel2.json",
            youtube_info_file="youtube_info_channel2.json"
        ),
        3: ChannelConfig(
            name="Channel 3",
            voice_id="UgBBYS2sOqTuMpoF3BR0",
            #BLaQKPB2UVQ1JfmZQYQn
            youtube_credentials_file="youtube_token_channel3.json",
            youtube_info_file="youtube_info_channel3.json"
        )
    })
    
    # Default channel to use if not specified
    default_channel: int = Field(default=1)
    
    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore"  # Ignore extra fields from environment variables
    }

# Create a global config instance
config = AppConfig()

# Helper function to get channel-specific config
def get_channel_config(channel_number: Optional[int] = None) -> ChannelConfig:
    """Get configuration for a specific channel"""
    channel = channel_number or config.default_channel
    if channel in config.channels:
        return config.channels[channel]
    raise ValueError(f"Invalid channel number: {channel}")

# Helper function to get timeline configuration with channel overrides applied
def get_timeline_config(channel_number: Optional[int] = None) -> TimelineConfig:
    """
    Get timeline configuration with channel-specific overrides applied.
    
    Args:
        channel_number: Channel number to get configuration for. If None,
                      uses the default channel.
                      
    Returns:
        TimelineConfig with channel-specific overrides applied
    """
    # Start with the global timeline config
    global_config = config.timeline
    
    # If no channel specified, return a copy of the global config
    if channel_number is None:
        # Create a copy of the global config
        import copy
        return copy.deepcopy(global_config)
    
    # Get the channel config
    try:
        channel_config = get_channel_config(channel_number)
    except ValueError:
        # Invalid channel, return a copy of the global config
        import copy
        return copy.deepcopy(global_config)
    
    # If channel has no timeline overrides, return a copy of the global config
    if channel_config.timeline is None:
        import copy
        return copy.deepcopy(global_config)
    
    # Create a copy of the global config
    import copy
    result = copy.deepcopy(global_config)
    
    # Apply channel-specific overrides
    channel_timeline = channel_config.timeline
    
    # Apply storage directory override
    if channel_timeline.storage_directory is not None:
        result.storage_directory = channel_timeline.storage_directory
        
    # Apply visualization directory override
    if channel_timeline.visualization_directory is not None:
        result.visualization_directory = channel_timeline.visualization_directory
        
    # Apply visualization detail level override
    if channel_timeline.default_detail_level is not None:
        result.visualization.default_detail_level = channel_timeline.default_detail_level
        
    # Apply timeline settings overrides
    if channel_timeline.default_framerate is not None:
        result.default_framerate = channel_timeline.default_framerate
        
    if channel_timeline.default_width is not None:
        result.default_width = channel_timeline.default_width
        
    if channel_timeline.default_height is not None:
        result.default_height = channel_timeline.default_height
        
    if channel_timeline.default_background is not None:
        result.default_background = channel_timeline.default_background
        
    # Apply export configuration overrides
    if channel_timeline.export is not None:
        channel_export = channel_timeline.export
        
        # General export settings
        if channel_export.export_directory is not None:
            result.export.export_directory = channel_export.export_directory
            
        if channel_export.default_format is not None:
            result.export.default_format = channel_export.default_format
            
        if channel_export.default_preset is not None:
            result.export.default_preset = channel_export.default_preset
            
        if channel_export.auto_analyze is not None:
            result.export.auto_analyze = channel_export.auto_analyze
            
        if channel_export.validate_before_export is not None:
            result.export.validate_before_export = channel_export.validate_before_export
            
        if channel_export.on_incompatible_elements is not None:
            result.export.on_incompatible_elements = channel_export.on_incompatible_elements
            
        # Format-specific settings
        if channel_export.create_sidecar_files is not None:
            result.export.format_config.create_sidecar_files = channel_export.create_sidecar_files
            
        if channel_export.normalize_paths is not None:
            result.export.format_config.normalize_paths = channel_export.normalize_paths
            
        if channel_export.convert_custom_elements is not None:
            result.export.format_config.convert_custom_elements = channel_export.convert_custom_elements
            
        if channel_export.relative_paths is not None:
            result.export.format_config.relative_paths = channel_export.relative_paths
            
        if channel_export.resolve_media_paths is not None:
            result.export.format_config.resolve_media_paths = channel_export.resolve_media_paths
            
        if channel_export.media_search_paths is not None:
            result.export.format_config.media_search_paths = channel_export.media_search_paths
            
    # Apply rendering configuration overrides
    if channel_timeline.rendering_enabled is not None:
        result.rendering.enabled = channel_timeline.rendering_enabled
        
    if channel_timeline.prefer_direct_rendering is not None:
        result.rendering.prefer_direct_rendering = channel_timeline.prefer_direct_rendering
        
    if channel_timeline.compatibility_mode is not None:
        result.rendering.compatibility_mode = channel_timeline.compatibility_mode
        
    if channel_timeline.force_fallback is not None:
        result.rendering.force_fallback = channel_timeline.force_fallback
        
    # Apply performance monitoring overrides
    if channel_timeline.enable_performance_monitoring is not None:
        result.rendering.enable_performance_monitoring = channel_timeline.enable_performance_monitoring
        
    if channel_timeline.track_memory_usage is not None:
        result.rendering.track_memory_usage = channel_timeline.track_memory_usage
        
    if channel_timeline.save_performance_reports is not None:
        result.rendering.save_performance_reports = channel_timeline.save_performance_reports
        
    if channel_timeline.performance_output_dir is not None:
        result.rendering.performance_output_dir = channel_timeline.performance_output_dir
    
    return result

def get_export_config(channel_number: Optional[int] = None) -> ExportConfig:
    """
    Get export configuration with channel-specific overrides applied.
    
    This is a convenience function that returns just the export portion
    of the timeline configuration, with all channel-specific overrides applied.
    
    Args:
        channel_number: Channel number to get configuration for. If None,
                      uses the default channel.
                      
    Returns:
        ExportConfig with channel-specific overrides applied
    """
    timeline_config = get_timeline_config(channel_number)
    return timeline_config.export

def apply_export_preset(export_config: ExportConfig, preset_name: str) -> ExportConfig:
    """
    Apply a predefined export preset to an export configuration.
    
    Args:
        export_config: The export configuration to apply the preset to
        preset_name: The name of the preset to apply (default, compatibility, professional, minimal)
        
    Returns:
        A new ExportConfig with the preset applied
        
    Raises:
        ValueError: If the preset_name is invalid
    """
    # Create a copy of the export config
    import copy
    result = copy.deepcopy(export_config)
    
    # Get the preset
    if preset_name == "default" and hasattr(result.presets, "default"):
        preset = result.presets.default
    elif preset_name == "compatibility" and hasattr(result.presets, "compatibility"):
        preset = result.presets.compatibility
    elif preset_name == "professional" and hasattr(result.presets, "professional"):
        preset = result.presets.professional
    elif preset_name == "minimal" and hasattr(result.presets, "minimal"):
        preset = result.presets.minimal
    else:
        raise ValueError(f"Invalid export preset: {preset_name}")
    
    # Apply the preset to the format config
    for key, value in preset.items():
        if hasattr(result.format_config, key):
            setattr(result.format_config, key, value)
    
    return result

# Legacy helper functions that now use FileManager internally
def get_abs_path(rel_path: str) -> Path:
    """
    Convert a relative path to absolute path using FileManager.
    
    This is maintained for backward compatibility with existing code.
    New code should use FileManager.get_abs_path() directly.
    """
    return file_mgr.get_abs_path(rel_path)
    
def get_channel_output_path(channel_number: Optional[int] = None) -> Path:
    """
    Get the output directory path for a specific channel using FileManager.
    
    This is maintained for backward compatibility with existing code.
    New code should use FileManager.get_channel_output_path() directly.
    """
    return file_mgr.get_channel_output_path(channel_number or config.default_channel)

if __name__ == "__main__":
    # Print configuration for debugging
    print(f"Configuration loaded from {__file__}")
    print(f"Base directory: {BASE_DIR}")
    
    # Example of accessing configuration values
    print("\nOpenAI Configuration:")
    print(f"  Model: {config.openai.script_model}")
    
    print("\nElevenLabs Configuration:")
    print(f"  Default Voice ID: {config.elevenlabs.default_voice_id}")
    
    print("\nFile Paths:")
    print(f"  Script: {get_abs_path(config.file_paths.script_file)}")
    print(f"  Voice: {get_abs_path(config.file_paths.voice_file)}")
    
    print("\nTimeline Configuration:")
    print(f"  Storage Directory: {config.timeline.storage_directory}")
    print(f"  Default Framerate: {config.timeline.default_framerate}")
    print(f"  Default Resolution: {config.timeline.default_width}x{config.timeline.default_height}")
    print(f"  Visualization:")
    print(f"    Detail Level: {config.timeline.visualization.default_detail_level}")
    print(f"    Max Width: {config.timeline.visualization.max_width}")
    print(f"  Serialization:")
    print(f"    Format: {config.timeline.serialization.format}")
    print(f"    Compression: {config.timeline.serialization.compression}")
    print(f"    Auto Backup: {config.timeline.serialization.auto_backup}")
    
    # Example of using the get_timeline_config function
    channel_timeline = get_timeline_config(1)
    print(f"\nChannel 1 Timeline Configuration:")
    print(f"  Storage Directory: {channel_timeline.storage_directory}")
    print(f"  Visualization Detail Level: {channel_timeline.visualization.default_detail_level}")
    
    # Example of using the get_export_config function
    export_config = get_export_config(1)
    print(f"\nChannel 1 Export Configuration:")
    print(f"  Export Directory: {export_config.export_directory}")
    print(f"  Default Format: {export_config.default_format}")
    print(f"  Default Preset: {export_config.default_preset}")
    
    # Example of applying a preset
    professional_config = apply_export_preset(export_config, "professional")
    print(f"\nProfessional Export Preset:")
    print(f"  Create Sidecar Files: {professional_config.format_config.create_sidecar_files}")
    print(f"  FCP11 Version: {professional_config.format_config.fcp11_version}")
    print(f"  Use Compound Clips: {professional_config.format_config.fcp11_use_compound_clips}")
    
    print("\nChannel Configurations:")
    for channel_num, channel_config in config.channels.items():
        print(f"  Channel {channel_num}: {channel_config.name}")
        print(f"    Voice ID: {channel_config.voice_id}")
        print(f"    YouTube Credentials: {channel_config.youtube_credentials_file}")
        if channel_config.timeline:
            print(f"    Timeline Overrides: Yes")
        else:
            print(f"    Timeline Overrides: No")
        
    # Example of using FileManager directly
    print("\nFileManager Examples:")
    print(f"  Channel 1 Script Path: {file_mgr.get_script_path(1, 'script')}")
    print(f"  Channel 2 Voice Path: {file_mgr.get_audio_output_path(2, 'generated_voice')}")
    print(f"  Channel 3 Video Path: {file_mgr.get_video_output_path(3, 'final_output')}")