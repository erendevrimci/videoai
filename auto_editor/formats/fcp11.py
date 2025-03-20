from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

if TYPE_CHECKING:
    from collections.abc import Sequence
    from fractions import Fraction

    from auto_editor.ffwrapper import FileInfo
    from auto_editor.timeline import TlAudio, TlVideo, v3
    from auto_editor.utils.log import Log


"""
Export a FCPXML 11 file readable with Final Cut Pro 10.6.8 or later.

See docs here:
https://developer.apple.com/documentation/professional_video_applications/fcpxml_reference

"""


def get_colorspace(src) -> str:
    # See: https://developer.apple.com/documentation/professional_video_applications/fcpxml_reference/asset#3686496

    # Handle FileInfo objects with videos attribute
    if hasattr(src, 'videos') and src.videos:
        s = src.videos[0]
        if hasattr(s, 'pix_fmt') and s.pix_fmt == "rgb24":
            return "sRGB IEC61966-2.1"
        if hasattr(s, 'color_space') and s.color_space == 5:  # "bt470bg"
            return "5-1-6 (Rec. 601 PAL)"
        if hasattr(s, 'color_space') and s.color_space == 6:  # "smpte170m"
            return "6-1-6 (Rec. 601 NTSC)"
        if hasattr(s, 'color_primaries') and s.color_primaries == 9:  # "bt2020"
            # See: https://video.stackexchange.com/questions/22059/how-to-identify-hdr-video
            if hasattr(s, 'color_transfer') and s.color_transfer in {16, 18}:  # "smpte2084" "arib-std-b67"
                return "9-18-9 (Rec. 2020 HLG)"
            return "9-1-9 (Rec. 2020)"

    # Default to Rec. 709 for all other cases (including Path objects)
    return "1-1-1 (Rec. 709)"


def make_name(src, tb: Fraction) -> str:
    # Handle FileInfo objects with get_res method
    if hasattr(src, 'get_res'):
        try:
            height = src.get_res()[1]
            if height == 720 and tb == 30:
                return "FFVideoFormat720p30"
            if height == 720 and tb == 25:
                return "FFVideoFormat720p25"
        except:
            pass
    
    # Default for Path objects or if get_res fails
    return "FFVideoFormatRateUndefined"


def fcp11_write_xml(
    group_name: str, version: int, output: str, resolve: bool, tl: v3, log: Log
) -> None:
    def fraction(val: int) -> str:
        if val == 0:
            return "0s"
        return f"{val * tl.tb.denominator}/{tl.tb.numerator}s"

    # Use timeline source or first video clip source if available
    src = tl.src
    if src is None and tl.v and tl.v[0] and len(tl.v[0]) > 0:
        src = tl.v[0][0].src
    assert src is not None, "Timeline or clips must have a valid source"

    # Get project name from source path, handling different object types
    if hasattr(src, 'path') and hasattr(src.path, 'stem'):
        proj_name = src.path.stem
    else:
        from pathlib import Path
        proj_name = Path(str(src)).stem
        
    # Get duration
    if hasattr(src, 'duration'):
        src_dur = int(src.duration * tl.tb)
    else:
        # Use timeline duration if source duration not available
        src_dur = tl.out_len()
    tl_dur = src_dur if resolve else tl.out_len()

    if version == 11:
        ver_str = "1.11"
    elif version == 10:
        ver_str = "1.10"
    elif version == 6:
        ver_str = "1.6"
    elif version == 5:
        ver_str = "1.5"
    else:
        log.error(f"Unknown final cut pro version: {version}")
        return

    fcpxml = Element("fcpxml", version=ver_str)
    resources = SubElement(fcpxml, "resources")

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
        # Get asset name handling different object types
        if hasattr(one_src, 'path') and hasattr(one_src.path, 'stem'):
            asset_name = one_src.path.stem
        else:
            from pathlib import Path
            asset_name = Path(str(one_src)).stem
            
        # Check for videos and audios attributes
        has_videos = hasattr(one_src, 'videos') and one_src.videos
        has_audios = hasattr(one_src, 'audios') and one_src.audios
        
        # Get audio channels
        audio_channels = 2  # Default to stereo
        if has_audios:
            audio_channels = one_src.audios[0].channels
            
        r2 = SubElement(
            resources,
            "asset",
            id=f"r{i * 2 + 2}",
            name=asset_name,
            start="0s",
            hasVideo="1" if has_videos else "0",
            format=f"r{i * 2 + 1}",
            hasAudio="1" if has_audios else "0",
            audioSources="1",
            audioChannels=f"{audio_channels}",
            duration=fraction(tl_dur),
        )
        
        # Create URI from source path
        if hasattr(one_src, 'path') and hasattr(one_src.path, 'resolve'):
            src_uri = one_src.path.resolve().as_uri()
        else:
            from pathlib import Path
            src_uri = Path(str(one_src)).resolve().as_uri()
            
        # Use metadata tag instead of media-rep for better DTD compatibility
        metadata = SubElement(r2, "metadata")
        SubElement(metadata, "md", key="com.apple.proapps.originalSource", value=src_uri)

    lib = SubElement(fcpxml, "library")
    evt = SubElement(lib, "event", name=group_name)
    proj = SubElement(evt, "project", name=proj_name)
    # Determine audio layout based on source type
    if hasattr(src, 'audios') and src.audios and hasattr(src.audios[0], 'channels'):
        audio_layout = "mono" if src.audios[0].channels == 1 else "stereo"
    else:
        # Default to stereo for Path objects or if audio info not available
        audio_layout = "stereo"
        
    sequence = SubElement(
        proj,
        "sequence",
        format="r1",
        tcStart="0s",
        tcFormat="NDF",
        audioLayout=audio_layout,
        audioRate="44.1k" if tl.sr == 44100 else "48k",
    )
    spine = SubElement(sequence, "spine")

    def make_clip(ref: str, clip: TlVideo | TlAudio) -> None:
        clip_properties = {
            "name": proj_name,
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

    if tl.v and tl.v[0]:
        clips: Sequence[TlVideo | TlAudio] = cast(Any, tl.v[0])
    elif tl.a and tl.a[0]:
        clips = tl.a[0]
    else:
        clips = []

    all_refs: list[str] = ["r2"]
    if resolve:
        for i in range(1, len(tl.a)):
            all_refs.append(f"r{(i + 1) * 2}")

    for my_ref in reversed(all_refs):
        for clip in clips:
            make_clip(my_ref, clip)

    tree = ElementTree(fcpxml)
    indent(tree, space="\t", level=0)
    tree.write(output, xml_declaration=True, encoding="utf-8")
