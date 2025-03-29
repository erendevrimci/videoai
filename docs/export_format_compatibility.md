# Export Format Compatibility Guide

This document provides detailed information about the compatibility of VideoAI's timeline exports with various video editing applications.

## Supported Export Formats

VideoAI supports exporting timelines to the following formats:

1. **JSON** - Auto-Editor's native format
2. **FCP7 XML** - Final Cut Pro 7 XML format (compatible with Adobe Premiere Pro)
3. **FCPXML** - Final Cut Pro X XML format
4. **Shotcut MLT** - Shotcut/Kdenlive MLT format

## Compatibility Matrix

| Format | Compatible Applications | Feature Support |
|--------|-------------------------|-----------------|
| JSON | Auto-Editor | Complete (all VideoAI features preserved) |
| FCP7 XML | Final Cut Pro 7, Adobe Premiere Pro, DaVinci Resolve | Good (basic timeline structure, limited effects) |
| FCPXML | Final Cut Pro X, DaVinci Resolve | Very Good (roles, markers, compound clips) |
| Shotcut MLT | Shotcut, Kdenlive | Good (filters, transitions) |

## Format Details

### JSON Format

The JSON format is Auto-Editor's native format and preserves all VideoAI timeline features.

**Compatibility:**
- **Auto-Editor**: Full compatibility
- **VideoAI**: Full compatibility

**Features Preserved:**
- All timeline structure (tracks, clips, timing)
- All custom elements (TlText, etc.)
- All metadata
- Effect parameters and settings

**Limitations:**
- Not directly usable by professional video editing applications

**Usage Notes:**
- Best format for saving work in progress
- Use for interchange between VideoAI projects
- Includes complete metadata and can be fully reloaded

### FCP7 XML Format

The FCP7 XML format is widely supported by professional video editing applications and provides good interoperability.

**Compatibility:**
- **Final Cut Pro 7**: Full compatibility
- **Adobe Premiere Pro**: Very good compatibility
- **DaVinci Resolve**: Good compatibility
- **Avid Media Composer**: Limited compatibility via import plugins

**Features Preserved:**
- Basic timeline structure (video/audio tracks)
- Clip timing and order
- Basic transitions
- Basic effects (depends on application)
- Media references

**Limitations:**
- Custom elements like TlText are converted to basic graphics/placeholders
- Complex effects may not translate between applications
- Limited support for nested sequences in some applications
- No direct support for markers in some applications

**Usage Notes:**
- Use `.xml.videoai.json` sidecar file to preserve metadata
- Best format for Adobe Premiere Pro workflow
- Generally good balance of compatibility vs. features
- Media paths may need to be absolute for best compatibility

### FCPXML Format

The FCPXML format is designed for Final Cut Pro X and newer versions, offering more advanced features.

**Compatibility:**
- **Final Cut Pro X**: Full compatibility
- **DaVinci Resolve**: Very good compatibility
- **Other applications**: Limited compatibility

**Features Preserved:**
- Timeline structure with full track support
- Roles (audio/video organization)
- Markers and comments
- Compound clips
- Effect parameters (when supported)
- Media references with proper frame rates

**Limitations:**
- Custom elements must be converted to FCP equivalents
- Effects are specific to Final Cut Pro X
- Less compatible with other applications than FCP7 XML

**Usage Notes:**
- Best format for Final Cut Pro X workflow
- Use version 5+ for modern FCPX features
- Complex timelines preserve best in this format
- Use `.fcpxml.videoai.json` sidecar file to preserve metadata

### Shotcut MLT Format

The MLT format is specific to Shotcut and related applications like Kdenlive that use the MLT framework.

**Compatibility:**
- **Shotcut**: Full compatibility
- **Kdenlive**: Good compatibility
- **Other applications**: Not compatible

**Features Preserved:**
- Timeline structure
- Transitions
- Filters
- Basic effects
- Media references

**Limitations:**
- Very limited compatibility outside the MLT ecosystem
- Custom elements must be converted to MLT-compatible elements
- Some VideoAI-specific features have no direct equivalent

**Usage Notes:**
- Best for Shotcut/Kdenlive workflows
- Good choice for open-source video editing workflow
- Use `.mlt.videoai.json` sidecar file to preserve metadata

## Import Testing Notes

Below are notes from testing imports of VideoAI exports in various applications:

### Final Cut Pro X

- FCPXML imports work best with version 5+ for modern FCPX
- Media paths should be resolvable (on the same system or using identical paths)
- Compound clips import correctly
- Roles are preserved
- Text elements are converted to basic text elements (formatting may be lost)
- Markers and comments are preserved

### Adobe Premiere Pro

- FCP7 XML provides the best compatibility
- Text elements convert to legacy title elements
- Basic transitions import correctly
- Some effects may not translate
- Media paths should be absolute for best results
- Multiple audio tracks are preserved
- Nested sequences may require adjustment

### DaVinci Resolve

- Supports both FCP7 XML and FCPXML
- FCPXML generally gives better results for complex timelines
- Media path resolution works well if media is accessible
- Text elements require manual adjustment
- Color information does not transfer
- Multi-track audio is preserved
- Compound clips may be flattened depending on settings

### Shotcut

- MLT format imports perfectly into Shotcut
- All filters and transitions are preserved
- Custom elements converted to basic MLT elements
- Media path resolution works well

## Tips for Best Results

1. **Media Files**:
   - Keep media files in locations accessible to the target application
   - Use absolute paths for maximum compatibility
   - Consider consolidating/collecting media before export

2. **Path Settings**:
   - For exchange between systems, use relative paths
   - For final export, use absolute paths
   - Set media search paths in the export configuration

3. **Format-Specific Settings**:
   - Use presets appropriate for your target application:
     - `professional`: Best for Final Cut Pro X
     - `compatibility`: Best for Adobe Premiere Pro
     - `default`: Good balance for most applications

4. **Custom Elements**:
   - Text elements will convert to basic equivalents (quality varies by format)
   - Complex custom elements may require manual adjustment after import
   - Consider simplifying complex elements when targeting other applications

5. **Metadata Preservation**:
   - Use sidecar files to preserve VideoAI-specific metadata
   - Sidecar files can be used to re-import back to VideoAI
   - JSON format preserves all metadata directly

## Known Issues and Workarounds

| Issue | Affected Format(s) | Workaround |
|-------|-------------------|------------|
| Text positioning mismatch | FCP7, FCPXML | Manually adjust text position after import |
| Missing media references | All | Use absolute paths or correct media paths in target app |
| Custom effects lost | All except JSON | Recreate effects in target application |
| Nested clip flattening | FCP7, Shotcut | Avoid complex nesting or recreate in target app |
| Audio track configuration | FCP7 | Manually adjust track configuration after import |

## Future Format Support

The export system is designed to be extensible. Future versions may add support for:

- BlackMagic DaVinci Resolve DRP format
- EDL (Edit Decision List) format
- AAF (Advanced Authoring Format)
- Other industry-standard interchange formats

## Testing Exports with Target Applications

To validate exports with target applications:

1. **Create test exports**: Use `test_export_with_target_apps.py` to generate test files
2. **Import into target application**: Open the exported file in your editing software
3. **Check for issues**: Verify timeline structure, media references, and custom elements
4. **Document results**: Update this compatibility guide with your findings

The test suite generates exports with various timeline features to help validate compatibility with different applications.