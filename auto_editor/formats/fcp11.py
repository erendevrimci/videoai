from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent
from pathlib import Path
import uuid

# Import directly - not just in TYPE_CHECKING
from collections.abc import Sequence
from fractions import Fraction
from auto_editor.ffwrapper import FileInfo
from auto_editor.timeline import TlAudio, TlVideo, v3
from auto_editor.utils.log import Log

# Try to import TlText from timeline_manager
try:
    from timeline_manager import TlText
    TLTEXT_IMPORTED = True
except ImportError:
    TLTEXT_IMPORTED = False
    # Define a placeholder class for type checking
    class TlText:
        pass


"""
Export a FCPXML 11 file readable with Final Cut Pro 10.6.8 or later.

See docs here:
https://developer.apple.com/documentation/professional_video_applications/fcpxml_reference

"""


def get_colorspace(src: FileInfo) -> str:
    # See: https://developer.apple.com/documentation/professional_video_applications/fcpxml_reference/asset#3686496

    if not src.videos:
        return "1-1-1 (Rec. 709)"

    s = src.videos[0]
    if s.pix_fmt == "rgb24":
        return "sRGB IEC61966-2.1"
    if s.color_space == 5:  # "bt470bg"
        return "5-1-6 (Rec. 601 PAL)"
    if s.color_space == 6:  # "smpte170m"
        return "6-1-6 (Rec. 601 NTSC)"
    if s.color_primaries == 9:  # "bt2020"
        # See: https://video.stackexchange.com/questions/22059/how-to-identify-hdr-video
        if s.color_transfer in {16, 18}:  # "smpte2084" "arib-std-b67"
            return "9-18-9 (Rec. 2020 HLG)"
        return "9-1-9 (Rec. 2020)"

    return "1-1-1 (Rec. 709)"


def convert_color(color_str: str) -> str:
    """
    Convert hex color to Final Cut Pro color format (R G B A)
    
    Args:
        color_str: Color in hex format '#RRGGBB' or '#RRGGBBAA'
        
    Returns:
        Color in FCP format 'R G B A' (values between 0-1)
    """
    if not color_str:
        return "1 1 1 1"  # Default white
        
    color_str = color_str.lstrip('#')
    
    if len(color_str) == 6:
        # No alpha specified, assume fully opaque
        r = int(color_str[0:2], 16) / 255.0
        g = int(color_str[2:4], 16) / 255.0
        b = int(color_str[4:6], 16) / 255.0
        a = 1.0
    elif len(color_str) == 8:
        # Alpha included in hex
        r = int(color_str[0:2], 16) / 255.0
        g = int(color_str[2:4], 16) / 255.0
        b = int(color_str[4:6], 16) / 255.0
        a = int(color_str[6:8], 16) / 255.0
    else:
        # Invalid format, return white
        return "1 1 1 1"
        
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"


def create_title_element(clip, parent_element, fraction_func, unique_id=None):
    """
    Create a title element for a TlText object
    
    Args:
        clip: The TlText object
        parent_element: The parent XML element to attach the title to
        fraction_func: Function to convert frames to FCPXML time format
        unique_id: Optional unique ID, will generate one if not provided
        
    Returns:
        The created title XML element
    """
    if not TLTEXT_IMPORTED or not isinstance(clip, TlText):
        return None
        
    # Generate a unique ID if not provided
    if unique_id is None:
        unique_id = str(uuid.uuid4()).replace('-', '')[:8]
        
    # Create <title> element with ref to built-in Basic Title effect
    title_element = SubElement(
        parent_element, 
        "title",
        ref="r5",  # Reference to Basic Title effect defined in resources
        name=f"{clip.text[:20]} - Basic Title",
        start="3600s",  # Standard offset for titles in FCP
        duration=fraction_func(clip.dur)
    )
    
    # Add basic title parameters
    param = SubElement(
        title_element, 
        "param", 
        name="Flatten", 
        key="9999/999166631/999166633/2/351", 
        value="1"
    )
    
    param = SubElement(
        title_element, 
        "param", 
        name="Alignment", 
        key="9999/999166631/999166633/2/354/999169573/401", 
        value="1 (Center)"
    )
    
    # Add text content with styling
    text_element = SubElement(title_element, "text")
    style_ref = f"ts{unique_id}"
    text_style = SubElement(text_element, "text-style", ref=style_ref)
    text_style.text = clip.text
    
    # Add text-style-def
    style_def = SubElement(title_element, "text-style-def", id=style_ref)
    text_style_element = SubElement(style_def, "text-style")
    
    # Apply text style properties
    text_style_element.set("font", clip.font)
    text_style_element.set("fontSize", str(clip.font_size))
    text_style_element.set("fontFace", "Regular")
    text_style_element.set("fontColor", convert_color(clip.color))
    
    # Handle background color as stroke with negative width (creates a background)
    if clip.bg_color:
        text_style_element.set("strokeColor", convert_color(clip.bg_color))
        text_style_element.set("strokeWidth", "-15")  # Negative for background
    
    # Standard baseline value for centered text
    text_style_element.set("baseline", "-229.1")
    
    # Set text alignment
    if clip.align == "left":
        text_style_element.set("alignment", "left")
    elif clip.align == "right":
        text_style_element.set("alignment", "right")
    else:
        text_style_element.set("alignment", "center")
    
    # Add shadow for better visibility if no background color is specified
    if not clip.bg_color:
        text_style_element.set("shadowColor", "0 0 0 0.75")
        text_style_element.set("shadowOffset", "5 315")
        text_style_element.set("shadowBlurRadius", "20")
    
    return title_element


def make_name(src: FileInfo, tb: Fraction) -> str:
    if src.get_res()[1] == 720 and tb == 30:
        return "FFVideoFormat720p30"
    if src.get_res()[1] == 720 and tb == 25:
        return "FFVideoFormat720p25"
    return "FFVideoFormatRateUndefined"


def fcp11_write_xml(
    group_name: str, version: int, output: str, resolve: bool, tl: v3, log: Log
) -> None:
    def fraction(val: int) -> str:
        if val == 0:
            return "0s"
        return f"{val * tl.tb.denominator}/{tl.tb.numerator}s"

    src = tl.src
    assert src is not None
    import json 
    with open("topics_covered.json") as f:
        json_data = json.load(f)
    proj_name = json_data.get('topics_already_covered', [])[-1] if json_data.get('topics_already_covered') else ""
    src_dur = int(src.duration * tl.tb)
    tl_dur = src_dur if resolve else tl.out_len()

    if version == 11:
        ver_str = "1.11"
    elif version == 10:
        ver_str = "1.10"
    else:
        log.error(f"Unknown final cut pro version: {version}")

    fcpxml = Element("fcpxml", version=ver_str)
    resources = SubElement(fcpxml, "resources")
    
    # Check if timeline contains any TlText objects
    has_text_objects = False
    if TLTEXT_IMPORTED:
        for track in tl.v:
            for clip in track:
                if isinstance(clip, TlText):
                    has_text_objects = True
                    break
            if has_text_objects:
                break
    
    # Add Basic Title effect resource if text objects are present
    if has_text_objects:
        SubElement(
            resources,
            "effect",
            id="r5",
            name="Basic Title",
            uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"
        )

    for i, one_src in enumerate(tl.unique_sources()):
        SubElement(
            resources,
            "format",
            id=f"r{i * 2 + 1}",
            name=make_name(one_src, tl.tb),
            frameDuration=fraction(1),
            width=f"{tl.res[0]}",
            height=f"{tl.res[1]}",
            colorSpace=get_colorspace(one_src),
        )
        r2 = SubElement(
            resources,
            "asset",
            id=f"r{i * 2 + 2}",
            name=one_src.path.stem,
            start="0s",
            hasVideo="1" if one_src.videos else "0",
            format=f"r{i * 2 + 1}",
            hasAudio="1" if one_src.audios else "0",
            audioSources="1",
            audioChannels=f"{2 if not one_src.audios else one_src.audios[0].channels}",
            duration=fraction(tl_dur),
        )
        SubElement(
            r2, "media-rep", kind="original-media", src=one_src.path.resolve().as_uri()
        )

    lib = SubElement(fcpxml, "library")
    evt = SubElement(lib, "event", name=group_name)
   
    

    
    proj = SubElement(evt, "project", name=proj_name)
    sequence = SubElement(
        proj,
        "sequence",
        format="r1",
        tcStart="0s",
        tcFormat="NDF",
        audioLayout="mono" if src.audios and src.audios[0].channels == 1 else "stereo",
        audioRate="44.1k" if tl.sr == 44100 else "48k",
    )
    spine = SubElement(sequence, "spine")

    def make_clip(ref: str, clip: TlVideo | TlAudio) -> None:
        # For the name property, use the appropriate filename from the asset reference
        # Find the asset with this reference
        asset_name = proj_name
        for resource in resources.findall(".//asset"):
            if resource.get("id") == ref:
                asset_name = resource.get("name")
                break
                
        clip_properties = {
            "name": asset_name,
            "ref": ref,
            "offset": fraction(clip.start),
            "duration": fraction(clip.dur),
            "start": fraction(clip.offset),
            "tcFormat": "NDF",
        }
        asset = SubElement(spine, "asset-clip", clip_properties)
        if clip.speed != 1:
            # See the "Time Maps" section.
            # https://developer.apple.com/documentation/professional_video_applications/fcpxml_reference/story_elements/timemap/

            timemap = SubElement(asset, "timeMap")
            SubElement(timemap, "timept", time="0s", value="0s", interp="smooth2")
            SubElement(
                timemap,
                "timept",
                time=fraction(int(src_dur // clip.speed)),
                value=fraction(src_dur),
                interp="smooth2",
            )

    # Generate the appropriate reference mapping for video clips
    video_clips = []
    video_refs = []
    text_clips = []  # Store text clips separately
    
    # Process all video tracks, including text tracks
    if tl.v:
        # Process main video track (track 0)
        if len(tl.v) > 0:
            for clip in tl.v[0]:
                if isinstance(clip, TlVideo) and hasattr(clip, 'src'):
                    # Find the corresponding asset reference
                    src_name = Path(clip.src.path).stem if isinstance(clip.src, FileInfo) else Path(str(clip.src)).stem
                    for i, one_src in enumerate(tl.unique_sources()):
                        if one_src.path.stem == src_name:
                            video_refs.append(f"r{i * 2 + 2}")
                            video_clips.append(clip)
                            break
        
        # Process additional tracks (including subtitle/text tracks)
        for track_idx in range(1, len(tl.v)):
            for clip in tl.v[track_idx]:
                if TLTEXT_IMPORTED and isinstance(clip, TlText):
                    # Store text clips to add them as titles later
                    text_clips.append((track_idx, clip))
    
    # Generate the appropriate reference mapping for audio clips
    audio_clips = []
    audio_refs = []
    
    if tl.a and tl.a[0]:
        for clip in tl.a[0]:
            if isinstance(clip, TlAudio) and hasattr(clip, 'src'):
                # Find the corresponding asset reference
                src_name = Path(clip.src.path).stem if isinstance(clip.src, FileInfo) else Path(str(clip.src)).stem
                for i, one_src in enumerate(tl.unique_sources()):
                    if one_src.path.stem == src_name:
                        audio_refs.append(f"r{i * 2 + 2}")
                        audio_clips.append(clip)
                        break
    
    # First add video clips to spine with correct references
    if len(video_clips) > 0 and len(video_refs) > 0:
        # Create the first video clip separately to hold the audio tracks
        first_video_clip = video_clips[0]
        first_video_ref = video_refs[0]
        
        # For the name property, use the appropriate filename from the asset reference
        # Find the asset with this reference
        asset_name = proj_name
        for resource in resources.findall(".//asset"):
            if resource.get("id") == first_video_ref:
                asset_name = resource.get("name")
                break
                
        # Create the parent video clip
        clip_properties = {
            "name": asset_name,
            "ref": first_video_ref,
            "offset": fraction(first_video_clip.start),
            "duration": fraction(first_video_clip.dur),
            "start": fraction(first_video_clip.offset),
            "tcFormat": "NDF",
        }
        parent_asset = SubElement(spine, "asset-clip", clip_properties)
        
        # Add audio clips nested inside the first video clip with lane="-1"
        for i, clip in enumerate(audio_clips):
            if i < len(audio_refs):
                audio_ref = audio_refs[i]
                # Find the asset name for this audio reference
                audio_name = proj_name
                for resource in resources.findall(".//asset"):
                    if resource.get("id") == audio_ref:
                        audio_name = resource.get("name")
                        break
                
                audio_properties = {
                    "name": audio_name,
                    "ref": audio_ref,
                    "offset": fraction(clip.start),
                    "duration": fraction(clip.dur),
                    "start": fraction(clip.offset),
                    "tcFormat": "NDF",
                    "lane": "-1",  # Set lane to -1 for audio tracks
                    "audioRole": "dialogue"  # Add audio role for better organization
                }
                audio_asset = SubElement(parent_asset, "asset-clip", audio_properties)
                
                # Add time map if needed
                if clip.speed != 1:
                    timemap = SubElement(audio_asset, "timeMap")
                    SubElement(timemap, "timept", time="0s", value="0s", interp="smooth2")
                    SubElement(
                        timemap,
                        "timept",
                        time=fraction(int(src_dur // clip.speed)),
                        value=fraction(src_dur),
                        interp="smooth2",
                    )
        
        # Add text clips as titles inside the first video clip
        for track_idx, text_clip in text_clips:
            # Set lane based on track index (ensures text appears above video)
            lane_value = str(track_idx)
            
            # Create unique ID for this title
            unique_id = str(uuid.uuid4()).replace('-', '')[:8]
            
            # Create title element
            title = create_title_element(text_clip, parent_asset, fraction, unique_id)
            
            # Set lane for the title (positive for overlay)
            if title is not None:
                title.set("lane", lane_value)
                title.set("offset", fraction(text_clip.start))
        
        # Add remaining video clips to spine
        for i in range(1, len(video_clips)):
            if i < len(video_refs):
                clip = video_clips[i]
                ref = video_refs[i]
                make_clip(ref, clip)
    else:
        # If there are no video clips, just add audio clips directly to spine
        for i, clip in enumerate(audio_clips):
            if i < len(audio_refs):
                # For the name property, use the appropriate filename from the asset reference
                # Find the asset with this reference
                asset_name = proj_name
                for resource in resources.findall(".//asset"):
                    if resource.get("id") == audio_refs[i]:
                        asset_name = resource.get("name")
                        break
                        
                clip_properties = {
                    "name": asset_name,
                    "ref": audio_refs[i],
                    "offset": fraction(clip.start),
                    "duration": fraction(clip.dur),
                    "start": fraction(clip.offset),
                    "tcFormat": "NDF",
                }
                asset = SubElement(spine, "asset-clip", clip_properties)

    tree = ElementTree(fcpxml)
    indent(tree, space="\t", level=0)
    tree.write(output, xml_declaration=True, encoding="utf-8")
