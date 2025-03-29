# Timeline Adapters

The Timeline Adapters module provides a flexible system for converting VideoAI timeline structures to various professional video editing formats. This is a key component of the export system, serving as the bridge between VideoAI's internal timeline representations and the formats expected by professional video editing software.

## Adapter Architecture

The adapter system follows a hierarchical design:

1. **TimelineAdapter (Base Class)**
   - Common functionality shared by all adapters
   - Path normalization
   - Metadata preservation
   - Media file collection

2. **Format-Specific Adapters**
   - **JSONAdapter**: Handles JSON format export
   - **ProfessionalFormatAdapter**: Base class for professional formats
   - **FCP7Adapter**: Final Cut Pro 7 / Premiere Pro XML format
   - **FCP11Adapter**: Final Cut Pro X FCPXML format
   - **ShotcutAdapter**: Shotcut MLT format

3. **AdapterFactory**
   - Factory pattern for creating the appropriate adapter

## Core Features

### Path Normalization

All adapters handle path normalization, ensuring that file paths in timelines are properly resolved for cross-platform compatibility.

```python
from timeline_adapters import TimelineAdapter

adapter = TimelineAdapter()
normalized_timeline = adapter._normalize_paths(original_timeline)
```

### Metadata Preservation

Adapters preserve VideoAI-specific metadata across different formats, which is essential for round-trip workflows.

```python
from timeline_adapters import TimelineAdapter

adapter = TimelineAdapter()
metadata = adapter.preserve_metadata(timeline, "fcp7")
```

### Custom Element Handling

Professional format adapters handle VideoAI-specific elements like `TlText` by converting them to compatible alternatives.

```python
from timeline_adapters import ProfessionalFormatAdapter

adapter = ProfessionalFormatAdapter()
converted_timeline = adapter._convert_custom_elements(timeline)
```

### Format-Specific Adaptations

Each format adapter implements specific adaptations required for their target format.

```python
from timeline_adapters import FCP11Adapter

adapter = FCP11Adapter()
adapted_timeline, metadata = adapter.adapt_timeline(timeline, "fcp11")
```

## Usage

### Basic Usage

```python
from timeline_adapters import AdapterFactory

# Create an adapter for a specific format
adapter = AdapterFactory.create_adapter("fcp7")

# Adapt a timeline for export
adapted_timeline, metadata = adapter.adapt_timeline(timeline)

# Get capabilities of all formats
from timeline_adapters import get_adapter_capabilities
capabilities = get_adapter_capabilities()
```

### Integration with Export Manager

The adapters are typically used through the ExportManager, which handles the full export process.

```python
from export_manager import ExportManager

export_mgr = ExportManager()
result_path = export_mgr.export_timeline(timeline, "fcp11")
```

## Format-Specific Considerations

### JSON Format

- Supports both v1 and v3 timelines
- Preserves all VideoAI metadata directly in the JSON structure
- Supports all custom elements without conversion

### FCP7 Format (Final Cut Pro 7 / Premiere Pro)

- Requires v3 timelines with video sources
- Custom elements are converted to rectangles as placeholders
- Metadata is stored in sidecar files (.videoai.json)
- Supports speed effects and transitions

### FCP11 Format (Final Cut Pro X)

- Requires v3 timelines with video sources
- Custom elements are converted to rectangles as placeholders
- Metadata is stored in sidecar files (.videoai.json)
- Supports roles, markers, and compound clips

### Shotcut Format (MLT)

- Requires v3 timelines with video sources
- Custom elements are converted to rectangles as placeholders
- Metadata is stored in sidecar files (.videoai.json)
- Supports filters and transitions

## Extending the System

### Adding New Adapters

To add support for a new export format:

1. Create a new adapter class inheriting from `TimelineAdapter` or `ProfessionalFormatAdapter`
2. Implement the `adapt_timeline` method
3. Add the new format to the `AdapterFactory`
4. Update the `get_adapter_capabilities` function

Example:

```python
class ResolveAdapter(ProfessionalFormatAdapter):
    """Adapter for DaVinci Resolve format export."""
    
    def adapt_timeline(self, timeline, format_type="resolve"):
        """Adapt a timeline for DaVinci Resolve export format."""
        # Call the base method for professional format adaptation
        timeline, metadata = super().adapt_timeline(timeline, format_type)
        
        # We know it's a v3 timeline at this point
        v3_timeline = cast(v3, timeline)
        
        # Add Resolve-specific metadata
        metadata["videoai_export"]["format_details"].update({
            "resolve_version": "18.0",
            "supports_fusion": True
        })
        
        return v3_timeline, metadata
```

Add to the factory:

```python
@staticmethod
def create_adapter(format_type, log=None):
    # ... existing code ...
    elif format_type == "resolve":
        return ResolveAdapter(log)
    # ... existing code ...
```

Update capabilities:

```python
def get_adapter_capabilities():
    capabilities = {
        # ... existing formats ...
        "resolve": {
            "supports_v1": False,
            "supports_v3": True,
            "supports_custom_elements": False,
            "supports_metadata": False,
            "description": "DaVinci Resolve format",
            "special_features": ["fusion", "color_grading"]
        }
    }
    return capabilities
```

## Implementation Details

### Timeline Validation

Adapters validate timelines to ensure they meet the requirements for each export format. This includes:

1. Timeline structure validation (v1 vs v3)
2. Media source validation
3. Custom element compatibility checks

### Sidecar Files

Professional format adapters create sidecar files (.videoai.json) to store VideoAI-specific metadata that can't be embedded in the export format directly.

### Error Handling

Adapters provide detailed error messages when a timeline can't be adapted for a specific format, helping users understand and resolve compatibility issues.

## Future Enhancements

1. **Native Text Support**: Better conversion of TlText elements to native text elements in professional formats
2. **Incremental Conversion**: Support for incremental adaptation of large timelines
3. **Cache Support**: Caching of adapter results for better performance with large timelines
4. **Media Management**: Enhanced media path handling for portability across systems