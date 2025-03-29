"""
Generate test timelines for export testing.

This script creates a set of test timelines with various features to test export
compatibility with different target applications.

Usage:
    python -m tests.generate_test_timelines

This will generate several test timelines in the tests/test_output directory.
"""
import os
import sys
import json
import tempfile
from pathlib import Path
from fractions import Fraction
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import VideoAI components
from timeline_manager import TimelineManager, TlText
from export_manager import ExportManager, FORMAT_JSON, FORMAT_FCP7, FORMAT_FCP11, FORMAT_SHOTCUT
from file_manager import FileManager
from auto_editor.timeline import v1, v3, TlVideo, TlAudio, TlImage, TlRect
from auto_editor.ffwrapper import initFileInfo

# Initialize managers
file_mgr = FileManager()
timeline_mgr = TimelineManager()
export_mgr = ExportManager()

# Create test output directory
TEST_OUTPUT_DIR = Path("tests/test_output")
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Mock FileInfo to avoid needing real media files
def mock_init_file_info(file_path, *args, **kwargs):
    """Create a mock FileInfo for the given file path."""
    mock_info = MagicMock()
    mock_info.path = Path(file_path)
    
    # Set up video properties
    mock_info.video = MagicMock()
    mock_info.video.width = 1920
    mock_info.video.height = 1080
    mock_info.video.duration = 60.0
    mock_info.video.fps = 30.0
    mock_info.video.rotation = 0
    
    # Set up audio properties
    mock_info.audio = MagicMock()
    mock_info.audio.samplerate = 48000
    mock_info.audio.channels = 2
    mock_info.audio.duration = 60.0
    
    return mock_info

# Apply the mock
patch('auto_editor.ffwrapper.initFileInfo', side_effect=mock_init_file_info).start()

def create_basic_timeline():
    """Create a basic timeline with minimal elements."""
    # Create a v3 timeline with standard properties
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add a video clip
    video_clip = TlVideo(
        start=0,
        dur=300,  # 10 seconds at 30fps
        src=Path("/mock/path/test_video_1.mp4")
    )
    timeline.v[0].append(video_clip)
    
    # Add an audio clip
    audio_clip = TlAudio(
        start=0,
        dur=300,
        src=Path("/mock/path/test_audio_1.wav")
    )
    timeline.a[0].append(audio_clip)
    
    # Add videoai_metadata
    timeline.videoai_metadata = {
        "version": "1.0",
        "type": "v3",
        "description": "Basic test timeline",
        "creator": "generate_test_timelines.py",
        "created_at": datetime.now().isoformat()
    }
    
    return timeline

def create_complex_timeline():
    """Create a complex timeline with various elements and effects."""
    # Create a v3 timeline with standard properties
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add multiple video clips on track 0
    video_clip1 = TlVideo(
        start=0,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/test_video_1.mp4")
    )
    timeline.v[0].append(video_clip1)
    
    video_clip2 = TlVideo(
        start=150,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/test_video_2.mp4")
    )
    timeline.v[0].append(video_clip2)
    
    # Add a second video track with another clip
    timeline.v.append([])  # Add track 1
    video_clip3 = TlVideo(
        start=75,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/test_video_3.mp4"),
        x=480,  # Position for picture-in-picture
        y=270,
        width=960,  # Half size for PiP
        height=540
    )
    timeline.v[1].append(video_clip3)
    
    # Add a text element (title)
    title_text = TlText(
        start=0,
        dur=90,  # 3 seconds at 30fps
        text="Complex Timeline Test",
        x=960,
        y=200,
        font="Arial",
        font_size=48,
        color="#FFFFFF"
    )
    timeline.v[0].append(title_text)
    
    # Add a subtitle text element
    subtitle_text = TlText(
        start=90,
        dur=90,  # 3 seconds at 30fps
        text="Testing Export Compatibility",
        x=960,
        y=280,
        font="Arial",
        font_size=36,
        color="#CCCCCC"
    )
    timeline.v[0].append(subtitle_text)
    
    # Add a rectangle overlay
    rect_overlay = TlRect(
        start=180,
        dur=120,  # 4 seconds at 30fps
        x=960,
        y=540,
        width=1600,
        height=200,
        fill="#3366CC",
        fill_opacity=0.7,
        rounded=20
    )
    timeline.v[0].append(rect_overlay)
    
    # Add multiple audio tracks
    audio_clip1 = TlAudio(
        start=0,
        dur=300,  # 10 seconds at 30fps
        src=Path("/mock/path/test_audio_1.wav")
    )
    timeline.a[0].append(audio_clip1)
    
    # Add a second audio track
    timeline.a.append([])  # Add track 1
    audio_clip2 = TlAudio(
        start=150,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/test_audio_2.wav"),
        volume=0.5  # Lower volume
    )
    timeline.a[1].append(audio_clip2)
    
    # Add a third audio track for music
    timeline.a.append([])  # Add track 2
    music_clip = TlAudio(
        start=0,
        dur=300,  # 10 seconds at 30fps
        src=Path("/mock/path/background_music.mp3"),
        volume=0.3  # Background volume
    )
    timeline.a[2].append(music_clip)
    
    # Add an image clip to the second video track
    image_clip = TlImage(
        start=225,
        dur=75,  # 2.5 seconds at 30fps
        src=Path("/mock/path/test_image.png"),
        x=1440,
        y=810,
        width=480,
        height=270
    )
    timeline.v[1].append(image_clip)
    
    # Add videoai_metadata with comprehensive information
    timeline.videoai_metadata = {
        "version": "1.0",
        "type": "v3",
        "description": "Complex test timeline with multiple tracks and elements",
        "creator": "generate_test_timelines.py",
        "created_at": datetime.now().isoformat(),
        "project": "Export Testing",
        "title": "Complex Timeline Export Test",
        "script": "This is a test script for the complex timeline export test.",
        "notes": "This timeline tests various features including:\n- Multiple video/audio tracks\n- Text elements\n- Rectangle overlays\n- Image elements\n- Picture-in-picture effects",
        "tags": ["test", "export", "complex", "multi-track"],
        "markers": [
            {"time": 0, "name": "Start", "comment": "Beginning of timeline"},
            {"time": 150, "name": "Middle", "comment": "Middle transition point"},
            {"time": 299, "name": "End", "comment": "End of timeline"}
        ]
    }
    
    return timeline

def create_long_timeline():
    """Create a longer timeline with many clips and crossfades."""
    # Create a v3 timeline with standard properties
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add many video clips with crossfades
    clip_duration = 90  # 3 seconds at 30fps
    fade_duration = 30  # 1 second fade
    num_clips = 10
    
    for i in range(num_clips):
        # Calculate start time with overlap for crossfades
        start_time = i * (clip_duration - fade_duration)
        
        # Video clip
        video_clip = TlVideo(
            start=start_time,
            dur=clip_duration,
            src=Path(f"/mock/path/test_video_{(i % 3) + 1}.mp4")
        )
        timeline.v[0].append(video_clip)
        
        # Audio clip
        audio_clip = TlAudio(
            start=start_time,
            dur=clip_duration,
            src=Path(f"/mock/path/test_audio_{(i % 2) + 1}.wav")
        )
        timeline.a[0].append(audio_clip)
    
    # Add titles throughout
    title_positions = [0, 270, 540]  # 0s, 9s, 18s
    
    for i, pos in enumerate(title_positions):
        title_text = TlText(
            start=pos,
            dur=90,  # 3 seconds at 30fps
            text=f"Title {i+1}",
            x=960,
            y=200 + (i * 70),
            font="Arial",
            font_size=48,
            color="#FFFFFF"
        )
        timeline.v[0].append(title_text)
    
    # Add background music for the full duration
    total_duration = num_clips * (clip_duration - fade_duration) + fade_duration
    
    # Add a second audio track for music
    timeline.a.append([])  # Add track 1
    music_clip = TlAudio(
        start=0,
        dur=total_duration,
        src=Path("/mock/path/background_music.mp3"),
        volume=0.2  # Background volume
    )
    timeline.a[1].append(music_clip)
    
    # Add videoai_metadata
    timeline.videoai_metadata = {
        "version": "1.0",
        "type": "v3",
        "description": "Long test timeline with crossfades",
        "creator": "generate_test_timelines.py",
        "created_at": datetime.now().isoformat(),
        "project": "Export Testing",
        "title": "Long Timeline with Crossfades",
        "tags": ["test", "export", "long", "crossfades"]
    }
    
    return timeline

def create_multicam_timeline():
    """Create a timeline that simulates a multi-camera setup."""
    # Create a v3 timeline with standard properties
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add three video tracks (one for each "camera")
    while len(timeline.v) < 3:
        timeline.v.append([])
    
    # Camera 1 (wide shot)
    cam1_clip = TlVideo(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/cam1_wide.mp4")
    )
    timeline.v[0].append(cam1_clip)
    
    # Camera 2 (medium shot)
    cam2_clip = TlVideo(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/cam2_medium.mp4")
    )
    timeline.v[1].append(cam2_clip)
    
    # Camera 3 (close-up)
    cam3_clip = TlVideo(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/cam3_closeup.mp4")
    )
    timeline.v[2].append(cam3_clip)
    
    # Add a fourth track for the active camera
    timeline.v.append([])
    
    # Switching between cameras
    # 0-5s: Camera 1
    active_cam1 = TlVideo(
        start=0,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/cam1_wide.mp4")
    )
    timeline.v[3].append(active_cam1)
    
    # 5-10s: Camera 2
    active_cam2 = TlVideo(
        start=150,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/cam2_medium.mp4")
    )
    timeline.v[3].append(active_cam2)
    
    # 10-15s: Camera 3
    active_cam3 = TlVideo(
        start=300,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/cam3_closeup.mp4")
    )
    timeline.v[3].append(active_cam3)
    
    # 15-20s: Back to Camera 1
    active_cam4 = TlVideo(
        start=450,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/cam1_wide.mp4")
    )
    timeline.v[3].append(active_cam4)
    
    # Add audio tracks
    # Main audio
    main_audio = TlAudio(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/main_audio.wav")
    )
    timeline.a[0].append(main_audio)
    
    # Add a second audio track for ambient sound
    timeline.a.append([])
    ambient_audio = TlAudio(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/ambient_audio.wav"),
        volume=0.3  # Lower volume
    )
    timeline.a[1].append(ambient_audio)
    
    # Add markers for camera switches
    markers = [
        {"time": 0, "name": "Start - Cam 1", "camera": 1},
        {"time": 150, "name": "Switch to Cam 2", "camera": 2},
        {"time": 300, "name": "Switch to Cam 3", "camera": 3},
        {"time": 450, "name": "Back to Cam 1", "camera": 1}
    ]
    
    # Add videoai_metadata
    timeline.videoai_metadata = {
        "version": "1.0",
        "type": "v3",
        "description": "Multi-camera setup test timeline",
        "creator": "generate_test_timelines.py",
        "created_at": datetime.now().isoformat(),
        "project": "Export Testing",
        "title": "Multi-Camera Setup",
        "multicam": True,
        "camera_count": 3,
        "markers": markers,
        "tags": ["test", "export", "multicam"]
    }
    
    return timeline

def create_effects_timeline():
    """Create a timeline with various effects and transitions."""
    # Create a v3 timeline with standard properties
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add video clips with various effects
    # Base video clip
    base_clip = TlVideo(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/base_clip.mp4")
    )
    timeline.v[0].append(base_clip)
    
    # Add text with animation effect
    animated_text = TlText(
        start=30,
        dur=120,  # 4 seconds at 30fps
        text="Animated Title",
        x=960,
        y=540,
        font="Arial",
        font_size=72,
        color="#FFFFFF",
        animation="fade-in",
        animation_duration=30  # 1 second animation
    )
    timeline.v[0].append(animated_text)
    
    # Add a second video track for overlays
    timeline.v.append([])
    
    # Add an image with zoom effect
    zoom_image = TlImage(
        start=150,
        dur=150,  # 5 seconds at 30fps
        src=Path("/mock/path/overlay_image.png"),
        x=960,
        y=540,
        width=960,
        height=540,
        scale_start=0.8,
        scale_end=1.2,
        animation="zoom"
    )
    timeline.v[1].append(zoom_image)
    
    # Add a rectangle with motion
    moving_rect = TlRect(
        start=300,
        dur=150,  # 5 seconds at 30fps
        x_start=200,
        x_end=1720,
        y_start=200,
        y_end=880,
        width=400,
        height=300,
        fill="#FF5500",
        fill_opacity=0.6,
        animation="linear-move"
    )
    timeline.v[1].append(moving_rect)
    
    # Add text that follows a path
    path_text = TlText(
        start=450,
        dur=150,  # 5 seconds at 30fps
        text="Following a Path",
        x_start=200,
        x_end=1720,
        y_start=880,
        y_end=200,
        font="Arial",
        font_size=48,
        color="#FFCC00",
        animation="path-follow",
        path_type="bezier"
    )
    timeline.v[1].append(path_text)
    
    # Add audio with fade effects
    fade_audio = TlAudio(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/main_audio.wav"),
        fade_in=60,  # 2 second fade in
        fade_out=60   # 2 second fade out
    )
    timeline.a[0].append(fade_audio)
    
    # Add a second audio track for music with volume adjustments
    timeline.a.append([])
    music_audio = TlAudio(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/music.mp3"),
        volume_start=0.1,
        volume_end=0.5,
        animation="volume-ramp"
    )
    timeline.a[1].append(music_audio)
    
    # Add videoai_metadata with effect details
    timeline.videoai_metadata = {
        "version": "1.0",
        "type": "v3",
        "description": "Timeline with various effects and transitions",
        "creator": "generate_test_timelines.py",
        "created_at": datetime.now().isoformat(),
        "project": "Export Testing",
        "title": "Effects and Transitions Test",
        "effects": [
            {"type": "fade-in", "element": "text", "time": 30},
            {"type": "zoom", "element": "image", "time": 150},
            {"type": "motion", "element": "rectangle", "time": 300},
            {"type": "path", "element": "text", "time": 450}
        ],
        "transitions": [
            {"type": "fade", "time": 0, "duration": 30},
            {"type": "cross-dissolve", "time": 150, "duration": 30},
            {"type": "wipe", "time": 300, "duration": 30},
            {"type": "fade", "time": 450, "duration": 30}
        ],
        "tags": ["test", "export", "effects", "transitions", "animation"]
    }
    
    return timeline

def create_nested_timeline():
    """Create a timeline with nested sequences/compound clips."""
    # Create a v3 timeline with standard properties
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Create a nested sequence for the intro
    intro_timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add elements to the intro timeline
    intro_video = TlVideo(
        start=0,
        dur=180,  # 6 seconds at 30fps
        src=Path("/mock/path/intro_video.mp4")
    )
    intro_timeline.v[0].append(intro_video)
    
    intro_text = TlText(
        start=30,
        dur=120,  # 4 seconds at 30fps
        text="Introduction",
        x=960,
        y=540,
        font="Arial",
        font_size=72,
        color="#FFFFFF"
    )
    intro_timeline.v[0].append(intro_text)
    
    intro_audio = TlAudio(
        start=0,
        dur=180,  # 6 seconds at 30fps
        src=Path("/mock/path/intro_audio.wav")
    )
    intro_timeline.a[0].append(intro_audio)
    
    # Create a nested sequence for the middle
    middle_timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add elements to the middle timeline
    middle_video1 = TlVideo(
        start=0,
        dur=120,  # 4 seconds at 30fps
        src=Path("/mock/path/middle_video1.mp4")
    )
    middle_timeline.v[0].append(middle_video1)
    
    middle_video2 = TlVideo(
        start=120,
        dur=120,  # 4 seconds at 30fps
        src=Path("/mock/path/middle_video2.mp4")
    )
    middle_timeline.v[0].append(middle_video2)
    
    middle_audio = TlAudio(
        start=0,
        dur=240,  # 8 seconds at 30fps
        src=Path("/mock/path/middle_audio.wav")
    )
    middle_timeline.a[0].append(middle_audio)
    
    # Create a nested sequence for the outro
    outro_timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=30,
        samplerate=48000
    )
    
    # Add elements to the outro timeline
    outro_video = TlVideo(
        start=0,
        dur=180,  # 6 seconds at 30fps
        src=Path("/mock/path/outro_video.mp4")
    )
    outro_timeline.v[0].append(outro_video)
    
    outro_text = TlText(
        start=30,
        dur=120,  # 4 seconds at 30fps
        text="Thank You For Watching",
        x=960,
        y=540,
        font="Arial",
        font_size=72,
        color="#FFFFFF"
    )
    outro_timeline.v[0].append(outro_text)
    
    outro_audio = TlAudio(
        start=0,
        dur=180,  # 6 seconds at 30fps
        src=Path("/mock/path/outro_audio.wav")
    )
    outro_timeline.a[0].append(outro_audio)
    
    # Add metadata to nested timelines
    intro_timeline.videoai_metadata = {"name": "Intro Sequence", "type": "nested"}
    middle_timeline.videoai_metadata = {"name": "Middle Sequence", "type": "nested"}
    outro_timeline.videoai_metadata = {"name": "Outro Sequence", "type": "nested"}
    
    # Now use these nested timelines in the main timeline
    # For demonstration purposes, we'll represent nested timelines as TlVideo objects
    # with special metadata indicating they're actually nested timelines
    
    # Add intro nested sequence
    intro_clip = TlVideo(
        start=0,
        dur=180,  # 6 seconds at 30fps
        src=Path("/mock/path/intro_nested.mp4")
    )
    intro_clip.nested_timeline = intro_timeline
    intro_clip.is_nested = True
    timeline.v[0].append(intro_clip)
    
    # Add middle nested sequence
    middle_clip = TlVideo(
        start=180,
        dur=240,  # 8 seconds at 30fps
        src=Path("/mock/path/middle_nested.mp4")
    )
    middle_clip.nested_timeline = middle_timeline
    middle_clip.is_nested = True
    timeline.v[0].append(middle_clip)
    
    # Add outro nested sequence
    outro_clip = TlVideo(
        start=420,
        dur=180,  # 6 seconds at 30fps
        src=Path("/mock/path/outro_nested.mp4")
    )
    outro_clip.nested_timeline = outro_timeline
    outro_clip.is_nested = True
    timeline.v[0].append(outro_clip)
    
    # Add corresponding audio for the full timeline
    main_audio = TlAudio(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/main_audio.wav"),
        volume=0.8
    )
    timeline.a[0].append(main_audio)
    
    # Add a second audio track for background music
    timeline.a.append([])
    bg_music = TlAudio(
        start=0,
        dur=600,  # 20 seconds at 30fps
        src=Path("/mock/path/background_music.mp3"),
        volume=0.2
    )
    timeline.a[1].append(bg_music)
    
    # Add videoai_metadata
    timeline.videoai_metadata = {
        "version": "1.0",
        "type": "v3",
        "description": "Timeline with nested sequences/compound clips",
        "creator": "generate_test_timelines.py",
        "created_at": datetime.now().isoformat(),
        "project": "Export Testing",
        "title": "Nested Sequences Test",
        "nested_sequences": [
            {"name": "Intro Sequence", "start": 0, "duration": 180},
            {"name": "Middle Sequence", "start": 180, "duration": 240},
            {"name": "Outro Sequence", "start": 420, "duration": 180}
        ],
        "tags": ["test", "export", "nested", "compound"]
    }
    
    return timeline

def export_all_formats(timeline, base_name):
    """Export a timeline to all supported formats."""
    formats = [
        (FORMAT_JSON, ".json"),
        (FORMAT_FCP7, ".xml"),
        (FORMAT_FCP11, ".fcpxml"),
        (FORMAT_SHOTCUT, ".mlt")
    ]
    
    results = {}
    
    for format_type, extension in formats:
        # Export with each preset
        presets = ["default", "compatibility", "professional", "minimal"]
        
        for preset in presets:
            # Create export manager with preset
            export_mgr_with_preset = ExportManager(export_preset=preset)
            
            # Export the timeline
            output_path = TEST_OUTPUT_DIR / f"{base_name}_{preset}_{format_type}{extension}"
            result = export_mgr_with_preset.export_timeline(timeline, format_type, output_path)
            
            if result:
                print(f"Exported {base_name} to {format_type} with {preset} preset: {output_path}")
                results[f"{format_type}_{preset}"] = str(output_path)
            else:
                print(f"Failed to export {base_name} to {format_type} with {preset} preset")
    
    return results

def main():
    """Generate test timelines and export to all formats."""
    print("Generating test timelines for export testing...")
    
    # Create and export basic timeline
    basic_timeline = create_basic_timeline()
    timeline_mgr.serialize_timeline(basic_timeline, TEST_OUTPUT_DIR / "basic_timeline.json")
    basic_exports = export_all_formats(basic_timeline, "basic")
    
    # Create and export complex timeline
    complex_timeline = create_complex_timeline()
    timeline_mgr.serialize_timeline(complex_timeline, TEST_OUTPUT_DIR / "complex_timeline.json")
    complex_exports = export_all_formats(complex_timeline, "complex")
    
    # Create and export long timeline
    long_timeline = create_long_timeline()
    timeline_mgr.serialize_timeline(long_timeline, TEST_OUTPUT_DIR / "long_timeline.json")
    long_exports = export_all_formats(long_timeline, "long")
    
    # Create and export multicam timeline
    multicam_timeline = create_multicam_timeline()
    timeline_mgr.serialize_timeline(multicam_timeline, TEST_OUTPUT_DIR / "multicam_timeline.json")
    multicam_exports = export_all_formats(multicam_timeline, "multicam")
    
    # Create and export effects timeline
    effects_timeline = create_effects_timeline()
    timeline_mgr.serialize_timeline(effects_timeline, TEST_OUTPUT_DIR / "effects_timeline.json")
    effects_exports = export_all_formats(effects_timeline, "effects")
    
    # Create and export nested timeline
    nested_timeline = create_nested_timeline()
    timeline_mgr.serialize_timeline(nested_timeline, TEST_OUTPUT_DIR / "nested_timeline.json")
    nested_exports = export_all_formats(nested_timeline, "nested")
    
    # Generate a summary report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "timelines": {
            "basic": {
                "path": str(TEST_OUTPUT_DIR / "basic_timeline.json"),
                "exports": basic_exports
            },
            "complex": {
                "path": str(TEST_OUTPUT_DIR / "complex_timeline.json"),
                "exports": complex_exports
            },
            "long": {
                "path": str(TEST_OUTPUT_DIR / "long_timeline.json"),
                "exports": long_exports
            },
            "multicam": {
                "path": str(TEST_OUTPUT_DIR / "multicam_timeline.json"),
                "exports": multicam_exports
            },
            "effects": {
                "path": str(TEST_OUTPUT_DIR / "effects_timeline.json"),
                "exports": effects_exports
            },
            "nested": {
                "path": str(TEST_OUTPUT_DIR / "nested_timeline.json"),
                "exports": nested_exports
            }
        },
        "formats": {
            "json": "Auto-Editor JSON format",
            "fcp7": "Final Cut Pro 7 XML format (also used by Adobe Premiere Pro)",
            "fcp11": "Final Cut Pro X XML format",
            "shotcut": "Shotcut MLT format"
        },
        "presets": {
            "default": "Balanced settings for general use",
            "compatibility": "Maximum compatibility with older software",
            "professional": "Best quality with modern features",
            "minimal": "Basic settings with minimal features"
        }
    }
    
    # Save summary report
    with open(TEST_OUTPUT_DIR / "test_exports_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTest timelines generated and exported to {TEST_OUTPUT_DIR}")
    print(f"Summary report saved to {TEST_OUTPUT_DIR / 'test_exports_summary.json'}")
    print("\nRecommended manual verification steps:")
    print("1. Import the exported files into their target applications")
    print("2. Verify timeline structure, clips, effects, and metadata")
    print("3. Document any compatibility issues in docs/export_format_compatibility.md")

if __name__ == "__main__":
    main()