# Export Verification Procedures

This document outlines the procedures for verifying the correctness and compatibility of timeline exports in various formats.

## Table of Contents

- [Overview](#overview)
- [General Verification Process](#general-verification-process)
- [JSON Format Verification](#json-format-verification)
- [FCP7 XML Format Verification](#fcp7-xml-format-verification)
- [FCPXML Format Verification](#fcpxml-format-verification)
- [Shotcut MLT Format Verification](#shotcut-mlt-format-verification)
- [Automated Verification](#automated-verification)
- [Manual Verification with Target Applications](#manual-verification-with-target-applications)

## Overview

Export verification ensures that:

1. Timelines export correctly to the chosen format
2. All timeline elements are preserved or properly converted
3. Media references are correctly maintained
4. Metadata is preserved (directly or via sidecar files)
5. Exported files can be imported into target applications

## General Verification Process

For any export format, follow these general verification steps:

1. **Preparation**:
   - Create a test timeline with representative elements
   - Include various clip types, tracks, and effects
   - Document expected elements and properties

2. **Export**:
   - Export the timeline to the desired format
   - Note any warnings or errors during export

3. **Basic Verification**:
   - Check that the output file exists and has appropriate size
   - Verify the file has the correct extension
   - Check for sidecar files if applicable

4. **Content Verification**:
   - Validate the file structure using the appropriate method for the format
   - Verify timeline elements are present
   - Check media references are correct
   - Validate metadata preservation

5. **Report**:
   - Document the verification results
   - Note any discrepancies or issues
   - Suggest improvements if needed

## JSON Format Verification

### File Structure Verification

1. **Parse the JSON file**:
   ```python
   import json
   with open(output_path, "r") as f:
       data = json.load(f)
   ```

2. **Verify timeline structure**:
   - Check `version` is correct (should be "3" for v3 timeline)
   - Check `timebase` is preserved (e.g., "30/1")
   - Verify `resolution` matches original (e.g., [1920, 1080])
   - Check `samplerate` matches original (e.g., 48000)

3. **Verify tracks**:
   - Count video tracks: `len(data["v"])`
   - Count audio tracks: `len(data["a"])`
   - Verify they match the original timeline

4. **Verify clips**:
   - Check each video/audio track contains the expected clips
   - Verify clip properties (start, duration, source path)
   - Check for special elements like TlText and verify properties

5. **Verify metadata**:
   - Check `videoai_metadata` is present
   - Verify all original metadata is preserved
   - Check for any export-specific metadata

### Reimport Verification

1. **Load the exported JSON back into VideoAI**:
   ```python
   loaded_timeline = timeline_mgr.deserialize_timeline(output_path)
   ```

2. **Verify timeline properties**:
   - Check resolution: `loaded_timeline.res`
   - Check framerate: `loaded_timeline.tb`
   - Check samplerate: `loaded_timeline.sr`

3. **Verify tracks and clips**:
   - Count video/audio tracks
   - Check clip count in each track
   - Verify clip properties

4. **Verify metadata**:
   - Check `videoai_metadata` is preserved
   - Verify all metadata fields match original

## FCP7 XML Format Verification

### File Structure Verification

1. **Parse the XML file**:
   ```python
   import xml.etree.ElementTree as ET
   tree = ET.parse(output_path)
   root = tree.getroot()
   ```

2. **Verify basic XML structure**:
   - Check root element is `xmeml`
   - Verify namespace if applicable
   - Check for `sequence` element

3. **Verify sequence properties**:
   - Check duration matches original timeline
   - Verify framerate: `sequence.find(".//timebase").text`
   - Check resolution: `sequence.find(".//width").text` and `sequence.find(".//height").text`

4. **Verify tracks**:
   - Count video tracks: `len(root.findall(".//track[@type='video']"))`
   - Count audio tracks: `len(root.findall(".//track[@type='audio']"))`
   - Verify they match original timeline

5. **Verify clips**:
   - Check each track contains the expected clips: `track.findall(".//clipitem")`
   - Verify clip properties (in, out, name)
   - Check for converted elements (text → graphics)

6. **Verify media references**:
   - Check file paths in `file/pathurl` elements
   - Verify media references use correct paths
   - Check for any missing references

7. **Check sidecar file**:
   - Verify `.xml.videoai.json` exists
   - Check metadata preservation
   - Verify export settings

### Adobe Premiere Pro Verification (Manual)

1. **Import the XML**:
   - Open Adobe Premiere Pro
   - Use File > Import to import the XML file
   - Note any import warnings or errors

2. **Verify sequence**:
   - Check sequence appears in Project panel
   - Open sequence in Timeline panel
   - Verify sequence settings match original (framerate, resolution)

3. **Verify tracks and clips**:
   - Count video/audio tracks
   - Check clip count and order in each track
   - Verify clip timing
   - Check for converted elements

4. **Verify media linking**:
   - Check if media is linked correctly
   - Note any offline media
   - Try relinking media if needed

5. **Test playback and export**:
   - Play the sequence to check timing
   - Check for rendering issues
   - Try exporting to verify full compatibility

## FCPXML Format Verification

### File Structure Verification

1. **Parse the XML file**:
   ```python
   import xml.etree.ElementTree as ET
   tree = ET.parse(output_path)
   root = tree.getroot()
   ```

2. **Verify basic XML structure**:
   - Check root element is `fcpxml`
   - Check version attribute: `root.attrib["version"]`
   - Verify `resources` and `library` elements

3. **Verify resources**:
   - Check format: `resources.find(".//format")`
   - Verify media assets: `resources.findall(".//asset")`
   - Check for proper media references

4. **Verify project/event structure**:
   - Check project element
   - Verify sequence structure
   - Check for proper nesting

5. **Verify timeline elements**:
   - Check for spine element
   - Verify clips in the timeline
   - Check for roles if applicable
   - Verify markers if present

6. **Verify media references**:
   - Check file paths in `asset` elements
   - Verify media references use correct paths
   - Check for any missing references

7. **Check sidecar file**:
   - Verify `.fcpxml.videoai.json` exists
   - Check metadata preservation
   - Verify export settings

### Final Cut Pro X Verification (Manual)

1. **Import the FCPXML**:
   - Open Final Cut Pro X
   - Use File > Import > XML to import the FCPXML file
   - Note any import warnings or errors

2. **Verify project/event**:
   - Check project appears in browser
   - Verify event structure
   - Check project settings match original

3. **Verify timeline**:
   - Open timeline in the timeline editor
   - Check clip count and order
   - Verify clip timing
   - Check for roles if applicable
   - Verify markers if present

4. **Verify media linking**:
   - Check if media is linked correctly
   - Note any offline media
   - Try relinking media if needed

5. **Test playback and export**:
   - Play the timeline to check timing
   - Check for rendering issues
   - Try exporting to verify full compatibility

## Shotcut MLT Format Verification

### File Structure Verification

1. **Parse the XML file**:
   ```python
   import xml.etree.ElementTree as ET
   tree = ET.parse(output_path)
   root = tree.getroot()
   ```

2. **Verify basic XML structure**:
   - Check root element is `mlt`
   - Check version attribute: `root.attrib["version"]`
   - Verify profile element: `root.find("profile")`

3. **Verify profile settings**:
   - Check resolution: `profile.attrib["width"]` and `profile.attrib["height"]`
   - Verify framerate: `profile.attrib["frame_rate_num"]` and `profile.attrib["frame_rate_den"]`
   - Check sample rate: `profile.attrib["sample_rate"]`

4. **Verify producers**:
   - Check for producer elements: `root.findall("producer")`
   - Verify producer properties (source, in/out points)
   - Check for proper media references

5. **Verify playlist structure**:
   - Check for playlist elements: `root.findall("playlist")`
   - Verify playlist entries: `playlist.findall("entry")`
   - Check for producer references

6. **Verify tractor and tracks**:
   - Check for tractor element (track container): `root.find("tractor")`
   - Verify track elements: `tractor.findall("track")`
   - Check track properties and references

7. **Verify filters and transitions**:
   - Check for filter elements: `root.findall(".//filter")`
   - Verify transition elements: `root.findall(".//transition")`
   - Check properties and parameters

8. **Check sidecar file**:
   - Verify `.mlt.videoai.json` exists
   - Check metadata preservation
   - Verify export settings

### Shotcut Verification (Manual)

1. **Import the MLT file**:
   - Open Shotcut
   - Use File > Open to open the MLT file
   - Note any import warnings or errors

2. **Verify project settings**:
   - Check Video Mode settings match original (resolution, framerate)
   - Verify audio settings match original

3. **Verify timeline**:
   - Check track count and types
   - Verify clip count and order in each track
   - Check clip timing
   - Verify filters and transitions

4. **Verify media linking**:
   - Check if media is linked correctly
   - Note any offline media
   - Try relinking media if needed

5. **Test playback and export**:
   - Play the timeline to check timing
   - Check for rendering issues
   - Try exporting to verify full compatibility

## Automated Verification

The test suite (`test_export_with_target_apps.py`) performs automated verification for all supported formats:

```bash
python -m unittest tests/test_export_with_target_apps.py
```

This test suite:
- Creates test timelines with various elements
- Exports to all supported formats
- Verifies file structure and content
- Checks for sidecar files and metadata preservation
- Documents compatibility notes

The test suite uses mock media files to avoid requiring real media files for testing.

## Manual Verification with Target Applications

For complete verification, manual testing with target applications is recommended:

1. **Generate test exports**:
   ```bash
   python -m tests.test_export_with_target_apps
   ```

2. **Import into target applications**:
   - Import generated test files into each target application
   - Follow the application-specific verification steps outlined above
   - Document any issues or unexpected behavior

3. **Create a verification report**:
   - Format: Pass/Fail for each verification step
   - Document specific issues with screenshots if possible
   - Note compatibility limitations
   - Suggest improvements for better compatibility

4. **Update documentation**:
   - Update the export_format_compatibility.md file with new findings
   - Document any workarounds for identified issues
   - Update verification procedures if needed