  Task 1: Enhance the FCP11 Exporter to Handle Text Objects

    - Modify /Volumes/drev-ventura/video-repos/videoai/auto_editor/formats/fcp11.py to process TlText objects
    - Implement a function to convert TlText objects to FCPXML title elements
    - Create proper FCPXML title formatting based on TlText properties

  Task 2: Create a Subtitle-Specific Converter

    - Implement a specialized function to convert subtitle text elements (identified by their track position or metadata)
    - Consider adding a flag in TlText or metadata to identify subtitle elements vs. regular text elements
    - Create style templates for subtitles that match the FCPXML format

  Task 3: Implement Subtitle Formatting Options

    - Create XML parameter values for font, size, color, alignment, etc.
    - Support background/highlight formatting options
    - Handle text styling (bold, italic) if supported by TlText

  Task 4: Integrate with Timeline Adapters

    - Update the TimelineAdapter class to recognize and process text objects
    - Add format-specific handling for text/subtitle elements in each format adapter

  Task 5: Testing and Validation

    - Create test cases with various subtitle formats and styles
    - Validate that subtitles appear correctly in FCP with proper timing
    - Ensure timing accuracy matches original SRT files
  3. Implementation Details

  For FCP11 XML Implementation:

  def create_title_element(clip, tl, parent_element):
      """Create a title element for a TlText object"""
      if isinstance(clip, TlText):
          # Create <title> element
          title_element = SubElement(parent_element, "title")
          title_element.set("ref", "title_ref_id")  # Generate unique ID
          title_element.set("duration", fraction(clip.dur))
          title_element.set("offset", fraction(clip.start))
          title_element.set("name", f"{clip.text[:20]} - Basic Title")

          # Add parameters for styling
          add_styling_parameters(title_element, clip)

          # Add text content
          text_element = SubElement(title_element, "text")
          text_style = SubElement(text_element, "text-style")
          text_style.set("ref", f"ts{unique_id}")  # Generate unique ID
          text_style.text = clip.text

          # Add text-style-def
          style_def = SubElement(title_element, "text-style-def")
          style_def.set("id", f"ts{unique_id}")
          text_style_element = SubElement(style_def, "text-style")

          # Set text style properties
          text_style_element.set("font", clip.font)
          text_style_element.set("fontSize", str(clip.font_size))
          text_style_element.set("fontColor", convert_color(clip.color))
          if clip.bg_color:
              text_style_element.set("strokeColor", convert_color(clip.bg_color))
              text_style_element.set("strokeWidth", "-15")  # Negative for background
          text_style_element.set("baseline", "-229.1")  # Based on examples
          text_style_element.set("alignment", clip.align)

  For TimelineAdapter Enhancement:

  def adapt_timeline_for_fcp11(self, timeline, format_type):
      """Adapt timeline with special handling for text objects"""
      # Handle standard elements
      # ...

      # Special handling for text/subtitle elements
      for track_idx, track in enumerate(timeline.v):
          for clip_idx, clip in enumerate(track):
              if isinstance(clip, TlText):
                  # Process text object for FCP11 format
                  # May need special handling if it's a subtitle (track_idx > 0)
                  # ...
  4. Best Practices and Considerations

    - Maintain Compatibility: Ensure changes don't break existing exports
    - Style Preservation: Make sure subtitle styling (fonts, colors) is preserved
    - Timing Accuracy: Ensure subtitle timing matches the SRT file exactly
    - Flexibility: Support multiple subtitle tracks if needed
    - Text Overflow: Handle long subtitles appropriately (wrapping, truncation)
    - Error Handling: Gracefully handle malformed SRT files or text objects
  5. Testing Approach

    - Create unit tests for the subtitle conversion functions
    - Test with various subtitle formats (SRT, VTT, etc.)
    - Test with different subtitle styles and formatting options
    - Verify that the generated FCPXML imports correctly into Final Cut Pro
    - Validate subtitle timing accuracy with the original SRT files


---

