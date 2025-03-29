# Timeline Configuration Guide

The VideoAI project provides a comprehensive configuration system for timelines, including export options. This document explains the available configuration options and how to use them.

## Configuration Structure

The configuration system is organized hierarchically:

- `AppConfig`: Root configuration for the entire application
  - `timeline`: Global timeline configuration (`TimelineConfig`)
    - `export`: Export-specific configuration (`ExportConfig`)
      - `format_config`: Format-specific export options (`FormatSpecificExportConfig`)
      - `presets`: Predefined export presets for common scenarios (`ExportPresets`)

## Channel-Specific Configuration

Each channel can override global configuration settings:

- `ChannelConfig`: Configuration for a specific channel
  - `timeline`: Channel-specific timeline configuration (`ChannelTimelineConfig`) 
    - `export`: Channel-specific export configuration (`ChannelExportConfig`)

## Export Configuration

The `ExportConfig` class provides settings for timeline exports:

```python
class ExportConfig(BaseModel):
    # Export directory for exported files
    export_directory: str = Field(default="exports")
    
    # Default export format
    default_format: str = Field(default="json", description="Default export format (json, fcp7, fcp11, shotcut)")
    
    # Default export preset
    default_preset: str = Field(default="default", description="Default export preset")
    
    # Format-specific configuration
    format_config: FormatSpecificExportConfig = Field(default_factory=FormatSpecificExportConfig)
    
    # Predefined export presets
    presets: ExportPresets = Field(default_factory=ExportPresets)
    
    # Auto-analyze timeline before export
    auto_analyze: bool = Field(default=True, description="Analyze timeline for export compatibility")
    
    # Default behavior for handling incompatible elements
    on_incompatible_elements: str = Field(
        default="convert",
        description="What to do with incompatible elements: 'convert', 'remove', 'error'"
    )
    
    # Validation rules
    validate_before_export: bool = Field(default=True, description="Validate timeline before export")
    max_validation_errors: int = Field(default=10, description="Maximum number of validation errors to report")
    stop_on_validation_error: bool = Field(default=False, description="Stop export on validation error")
```

## Format-Specific Configuration

The `FormatSpecificExportConfig` class provides format-specific export settings:

```python
class FormatSpecificExportConfig(BaseModel):
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
```

## Export Presets

The configuration system includes predefined presets for common export scenarios:

```python
class ExportPresets(BaseModel):
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
```

## Channel-Specific Export Overrides

Channels can override global export configuration:

```python
class ChannelExportConfig(BaseModel):
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
```

## Using Configuration in Channel Settings

Here's an example of configuring channel-specific export settings:

```python
# Configure Channel 1 with specific export settings
channel_1 = ChannelConfig(
    name="Channel 1",
    voice_id="UgBBYS2sOqTuMpoF3BR0",
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
)
```

## Accessing Configuration in Code

You can access the configuration system in various ways:

```python
# Import configuration helpers
from config import get_export_config, apply_export_preset, get_timeline_config

# Get export configuration for the default channel
export_config = get_export_config()

# Get export configuration for a specific channel
channel_export = get_export_config(channel_number=1)

# Apply a preset to the configuration
professional_config = apply_export_preset(export_config, "professional")

# Use in ExportManager
from export_manager import ExportManager

# Create export manager with channel and preset
export_mgr = ExportManager(channel_number=1, export_preset="professional")

# Apply custom format settings
export_mgr.set_format_config(
    fcp11_version=6,
    fcp11_use_compound_clips=True,
    relative_paths=False
)

# Export timeline with configuration
result = export_mgr.export_timeline(
    timeline="path/to/timeline.json",
    format_type="fcp11"  # Will use preset and configuration options
)
```

## Command Line Options

The export system provides command-line options for setting configuration values:

```bash
# Apply a preset
python export_manager.py --timeline path/to/timeline.json --preset professional

# Override specific settings
python export_manager.py --timeline path/to/timeline.json --format fcp11 --fcp11-version 6 --no-sidecar

# Show configuration
python export_manager.py --show-config

# Show available presets
python export_manager.py --list-presets
```

## Environment Variables

You can set configuration options through environment variables using the `VIDEOAI__TIMELINE__EXPORT__` prefix:

```bash
# Set default export format
export VIDEOAI__TIMELINE__EXPORT__DEFAULT_FORMAT="fcp11"

# Set format-specific options
export VIDEOAI__TIMELINE__EXPORT__FORMAT_CONFIG__FCP11_VERSION=6
```

## Configuration Files

The configuration system supports loading from `.env` files or directly from the Python configuration:

```python
# In config.py or a custom config file
timeline = TimelineConfig(
    storage_directory="custom_timelines",
    export=ExportConfig(
        default_format="fcp11",
        default_preset="professional",
        export_directory="custom_exports"
    )
)
```