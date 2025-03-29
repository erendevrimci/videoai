# Timeline Export System

The VideoAI Timeline Export System provides functionality to export timelines to various professional editing formats. This document explains how to use the export system and how to configure it for different export scenarios.

## Supported Export Formats

The Timeline Export System supports the following export formats:

- **JSON**: Auto-Editor JSON format for timeline interchange
- **FCP7**: Final Cut Pro 7 XML format (also used by Adobe Premiere Pro)
- **FCP11**: Final Cut Pro X XML format
- **Shotcut**: Shotcut MLT format

## Export Configuration

The export system is highly configurable through the configuration system in `config.py`. The main configuration class for exports is `ExportConfig`, which includes settings for all supported formats.

### Basic Export Settings

```python
# Get the export configuration for the default channel
from config import get_export_config
export_config = get_export_config()

# Get export configuration for a specific channel
channel_export = get_export_config(channel_number=1)

# Access basic export settings
export_directory = export_config.export_directory
default_format = export_config.default_format
default_preset = export_config.default_preset
```

### Format-Specific Settings

Each export format has specific settings that can be configured:

```python
# Access format-specific settings
format_config = export_config.format_config

# FCP7 settings
fcp7_version = format_config.fcp7_version
fcp7_use_markers = format_config.fcp7_use_markers

# FCP11 settings
fcp11_version = format_config.fcp11_version
fcp11_use_roles = format_config.fcp11_use_roles
fcp11_use_markers = format_config.fcp11_use_markers
fcp11_use_compound_clips = format_config.fcp11_use_compound_clips

# Shotcut settings
shotcut_version = format_config.shotcut_version
shotcut_use_filters = format_config.shotcut_use_filters

# Common settings
create_sidecar_files = format_config.create_sidecar_files
normalize_paths = format_config.normalize_paths
convert_custom_elements = format_config.convert_custom_elements
relative_paths = format_config.relative_paths
resolve_media_paths = format_config.resolve_media_paths
```

### Export Presets

The export system includes predefined presets for common export scenarios:

- **default**: Balanced settings for general use
- **compatibility**: Maximum compatibility with older software
- **professional**: Best quality with modern features
- **minimal**: Basic settings with minimal features

You can apply presets using the `apply_export_preset` function:

```python
from config import get_export_config, apply_export_preset

# Get the base export configuration
export_config = get_export_config()

# Apply a preset
professional_config = apply_export_preset(export_config, "professional")

# Use the preset configuration for export
# ...
```

### Channel-Specific Export Configuration

Each channel can override the global export configuration settings:

```python
from config import get_channel_config

# Get a channel configuration
channel_config = get_channel_config(1)

# Check if the channel has export overrides
if channel_config.timeline and channel_config.timeline.export:
    # Access channel-specific export settings
    channel_export = channel_config.timeline.export
    
    # Export format override
    if channel_export.default_format:
        print(f"Channel 1 uses {channel_export.default_format} format by default")
        
    # Export preset override
    if channel_export.default_preset:
        print(f"Channel 1 uses {channel_export.default_preset} preset by default")
```

## Using the Export Manager

The `ExportManager` class provides a simple interface for exporting timelines to different formats.

```python
from export_manager import ExportManager

# Create an export manager for a specific channel
export_mgr = ExportManager(channel_number=1)

# Export a timeline to JSON format
export_mgr.export_to_json(
    timeline="path/to/timeline.json",
    output_path="path/to/output.json"
)

# Export a timeline to FCP7 format
export_mgr.export_to_fcp7(
    timeline="path/to/timeline.json",
    output_path="path/to/output.xml"
)

# Export a timeline to FCP11 format
export_mgr.export_to_fcp11(
    timeline="path/to/timeline.json",
    output_path="path/to/output.fcpxml"
)

# Export a timeline to Shotcut format
export_mgr.export_to_shotcut(
    timeline="path/to/timeline.json",
    output_path="path/to/output.mlt"
)

# Generic export method
export_mgr.export_timeline(
    timeline="path/to/timeline.json",
    format_type="fcp11",
    output_path="path/to/output.fcpxml"
)
```

## Timeline Compatibility

Not all timelines are compatible with all export formats. You can check a timeline's compatibility with different formats:

```python
# Get information about a timeline for export compatibility
timeline_info = export_mgr.get_timeline_info("path/to/timeline.json")

# Check if the timeline is compatible with FCP11 format
if "fcp11" in timeline_info["compatible_formats"]:
    print("Timeline is compatible with FCP11 format")
else:
    print("Timeline is not compatible with FCP11 format")
    
# Check for format-specific warnings
if "format_warnings" in timeline_info and "fcp11" in timeline_info["format_warnings"]:
    print(f"Warning for FCP11 format: {timeline_info['format_warnings']['fcp11']}")
```

## Validation and Error Handling

The export system includes validation to ensure timelines meet the requirements for specific formats:

```python
# Configure validation settings
export_config.validate_before_export = True
export_config.max_validation_errors = 10
export_config.stop_on_validation_error = False

# Configure handling of incompatible elements
export_config.on_incompatible_elements = "convert"  # Options: "convert", "remove", "error"
```

## Configuration Reference

### Export Configuration (`ExportConfig`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| export_directory | str | "exports" | Directory for exported files |
| default_format | str | "json" | Default export format |
| default_preset | str | "default" | Default export preset |
| auto_analyze | bool | True | Analyze timeline before export |
| validate_before_export | bool | True | Validate timeline before export |
| max_validation_errors | int | 10 | Maximum validation errors to report |
| stop_on_validation_error | bool | False | Stop export on validation error |
| on_incompatible_elements | str | "convert" | How to handle incompatible elements |

### Format-Specific Configuration (`FormatSpecificExportConfig`)

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| create_sidecar_files | bool | True | Create sidecar files for metadata |
| normalize_paths | bool | True | Normalize media file paths |
| convert_custom_elements | bool | True | Convert custom elements |
| relative_paths | bool | True | Use relative paths when possible |
| resolve_media_paths | bool | True | Resolve missing media files |
| media_search_paths | List[str] | [] | Additional paths to search |
| fcp7_version | int | 5 | FCP7 XML schema version |
| fcp11_version | int | 5 | FCP11 XML schema version |
| shotcut_version | str | "7.0" | Shotcut MLT version |
| fcp7_use_markers | bool | True | Include markers in FCP7 exports |
| fcp11_use_roles | bool | True | Use roles in FCP11 exports |
| fcp11_use_markers | bool | True | Include markers in FCP11 exports |
| fcp11_use_compound_clips | bool | True | Use compound clips in FCP11 exports |
| shotcut_use_filters | bool | True | Include filters in Shotcut exports |

## Command Line Interface

The export system includes a command-line interface for exporting timelines:

```bash
# Export a timeline to JSON format
python export_manager.py --timeline path/to/timeline.json --format json --output path/to/output.json

# Export a timeline to FCP11 format for a specific channel
python export_manager.py --timeline path/to/timeline.json --format fcp11 --output path/to/output.fcpxml --channel 1

# Show information about a timeline
python export_manager.py --timeline path/to/timeline.json --info

# List supported formats
python export_manager.py --list-formats
```