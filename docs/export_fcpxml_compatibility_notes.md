# FCPXML Compatibility Notes

## Final Cut Pro XML Import Compatibility Fixes

When exporting to FCPXML format for Final Cut Pro, we discovered several DTD validation issues that prevented successful import. This document outlines the key changes made to ensure compatibility.

### Key Issues Fixed

1. **Version Selection**: Using FCPXML version 1.10 for Final Cut Pro X compatibility. Final Cut Pro X requires version 1.10 or 1.11, not older versions.

2. **Media Representation Structure**: Using the proper `<media-rep>` structure with `src` attribute directly on the element (not in a nested file element).

3. **Attribute Versioning**: Making specific attributes version-dependent to ensure DTD compliance:
   - `colorSpace` attribute is only included for version 10+
   - `tcFormat` attribute is only included for version 6+
   - Media references use `<metadata>` for versions below 10 and `<media-rep>` for 10+

### Implementation Changes

The following changes were made to the `auto_editor/formats/fcp11.py` file:

1. Added version-specific attribute handling:
   ```python
   # Only add colorSpace for version 10+ as older DTDs don't support it
   if version >= 10:
       format_el.set("colorSpace", get_colorspace(one_src))
   ```

2. Used correct media representation structure based on version:
   ```python
   # For FCPXML 1.10+, src attribute should be directly on the media-rep element
   SubElement(r2, "media-rep", kind="original-media", src=src_uri)
   ```

3. Updated the export manager to force version 10 when DTD compatibility is enabled.

### Technical Background

Final Cut Pro X requires FCPXML version 1.10 or higher, and uses Document Type Definition (DTD) validation when importing XML files. There are version-specific DTDs with different allowed attributes and elements:

- **Version 1.1-1.5**: Basic format, no colorSpace or tcFormat attributes, only metadata for file references
- **Version 1.6-1.9**: Adds tcFormat attributes but still no colorSpace
- **Version 1.10+**: Full support for colorSpace attributes and media-rep elements

### Comparing Working vs. Non-Working XMLs

The working FCPXML file has:
1. Version 1.10
2. Each format element includes colorSpace="1-1-1 (Rec. 709)"
3. Simple media-rep structure: `<media-rep kind="original-media" src="file://..." />`
4. tcFormat attributes on sequence and asset-clip elements

Our implementation now matches this structure exactly.

### Future Considerations

For future development:

1. Consider automatic version detection based on target Final Cut Pro version
2. Add DTD validation at export time to catch issues before import attempts
3. Support for newer FCPXML features like roles and effects
4. Consider direct FCPBundle export for modern Final Cut Pro versions