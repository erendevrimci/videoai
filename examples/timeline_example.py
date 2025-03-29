"""
Example usage of TimelineManager for VideoAI project.

This example demonstrates how to:
1. Create timelines from video clips
2. Convert VideoAI clip sequences to timelines
3. Serialize and deserialize timelines
4. Visualize timeline structure
5. Convert between timeline formats
6. Use advanced timeline serialization with metadata
"""
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import project modules
sys.path.append(str(Path(__file__).parent.parent))

from timeline_manager import TimelineManager
from file_manager import FileManager

# Initialize file manager and timeline manager
file_mgr = FileManager()
timeline_mgr = TimelineManager(channel_number=1)

def basic_timeline_example():
    """Basic timeline creation example"""
    print("\n=== Basic Timeline Creation ===")
    
    # Create a simple v3 timeline (empty)
    timeline = timeline_mgr.create_v3_timeline(
        width=1080,
        height=1920,
        framerate=30,
        samplerate=48000,
        background="#000000"
    )
    
    # Print timeline details
    print(f"Created timeline with resolution {timeline.res[0]}x{timeline.res[1]}")
    print(f"Timebase: {timeline.tb}")
    print(f"Sample rate: {timeline.sr}")
    
    # Visualize the timeline
    print("\nTimeline Visualization:")
    print(timeline_mgr.visualize_timeline(timeline))
    
    # Serialize the timeline
    timeline_dict = timeline_mgr.serialize_timeline(timeline)
    print(f"Timeline serialized to dictionary with {len(timeline_dict)} keys")
    
    # Save to file
    example_dir = file_mgr.get_abs_path("examples/output")
    file_mgr.ensure_dir_exists(example_dir)
    output_path = example_dir / "basic_timeline.json"
    timeline_mgr.serialize_timeline(timeline, output_path)
    print(f"Timeline saved to {output_path}")

def clip_sequence_example():
    """Example of converting a clip sequence to a timeline"""
    print("\n=== Clip Sequence Conversion ===")
    
    # Define a simple clip sequence (similar to what video_edit.py creates)
    clip_sequence = [
        {
            "clip_name": "sample_clips/placeholder.mp4",
            "start_time": 0,
            "duration": 5.0,
            "script_segment": "First segment of the script"
        },
        {
            "clip_name": "sample_clips/placeholder.mp4",
            "start_time": 2.0,
            "duration": 6.0,
            "script_segment": "Second segment of the script"
        }
    ]
    
    # Check if we have a sample video
    sample_path = file_mgr.get_abs_path("clips/sample_clips/placeholder.mp4")
    if not file_mgr.file_exists(sample_path):
        print(f"Sample video not found at {sample_path}")
        print("Creating placeholder clips directory...")
        
        clips_dir = file_mgr.get_abs_path("clips/sample_clips")
        file_mgr.ensure_dir_exists(clips_dir)
        
        print(f"This example requires a sample video at: {sample_path}")
        print("Please create this file or run video_edit.py which will create it.")
        return
    
    # Convert clip sequence to timeline
    timeline = timeline_mgr.clip_sequence_to_timeline(
        clip_sequence,
        output_width=1080,
        output_height=1920,
        framerate=30
    )
    
    # Print details of the generated timeline
    print(f"Converted {len(clip_sequence)} clips to timeline")
    
    if len(timeline.v[0]) > 0:
        print(f"Timeline has {len(timeline.v[0])} video clips on track 0")
        total_duration_frames = sum(clip.dur for clip in timeline.v[0])
        total_duration_sec = total_duration_frames / timeline.tb
        print(f"Total timeline duration: {total_duration_sec:.2f} seconds")
    else:
        print("Timeline has no video clips (conversion may have failed)")
    
    # Visualize the timeline
    print("\nTimeline Visualization:")
    print(timeline_mgr.visualize_timeline(timeline))
    
    # Save to file
    example_dir = file_mgr.get_abs_path("examples/output")
    file_mgr.ensure_dir_exists(example_dir)
    output_path = example_dir / "clip_sequence_timeline.json"
    timeline_mgr.serialize_timeline(timeline, output_path)
    print(f"Timeline saved to {output_path}")

def timeline_manipulation_example():
    """Example of creating and manipulating a timeline"""
    print("\n=== Timeline Manipulation ===")
    
    # Create a v3 timeline
    timeline = timeline_mgr.create_v3_timeline(
        width=1080,
        height=1920,
        framerate=30
    )
    
    # Check if we have a sample video
    sample_path = file_mgr.get_abs_path("clips/sample_clips/placeholder.mp4")
    if not file_mgr.file_exists(sample_path):
        print(f"Sample video not found at {sample_path}")
        print(f"This example requires a sample video at: {sample_path}")
        return
    
    # Initialize the video source
    from auto_editor.ffwrapper import initFileInfo
    from auto_editor.timeline import TlVideo
    
    # Get a source video
    src = initFileInfo(str(sample_path), timeline_mgr.log)
    
    # Create video objects for timeline
    video1 = TlVideo(
        start=0,  # Start at frame 0
        dur=90,   # 3 seconds at 30fps
        src=src,
        offset=0,  # Start at beginning of source
        speed=1.0,
        stream=0
    )
    
    video2 = TlVideo(
        start=90,  # Start at frame 90 (3 seconds at 30fps)
        dur=90,    # 3 seconds at 30fps
        src=src,
        offset=30,  # Start 1 second into source
        speed=1.0,
        stream=0
    )
    
    # Add clips to video track
    timeline.v[0].append(video1)
    timeline.v[0].append(video2)
    
    # Add an audio track if source has audio
    if hasattr(src, 'audio') and src.audio:
        from auto_editor.timeline import TlAudio
        
        audio1 = TlAudio(
            start=0,    # Start at frame 0
            dur=90,     # 3 seconds at 30fps
            src=src,
            offset=0,   # Start at beginning of source
            speed=1.0,
            volume=1.0,
            stream=0
        )
        
        audio2 = TlAudio(
            start=90,   # Start at frame 90
            dur=90,     # 3 seconds
            src=src,
            offset=30,  # Start 1 second into source
            speed=1.0,
            volume=0.8, # Slightly quieter
            stream=0
        )
        
        # Add to audio track
        timeline.a[0].append(audio1)
        timeline.a[0].append(audio2)
    
    # Print details
    print(f"Created timeline with {len(timeline.v[0])} video clips")
    if len(timeline.a) > 0 and len(timeline.a[0]) > 0:
        print(f"Timeline has {len(timeline.a[0])} audio clips")
    
    # Calculate total duration
    total_frames = timeline.end
    total_seconds = total_frames / timeline.tb
    print(f"Total timeline duration: {total_seconds:.2f} seconds ({total_frames} frames)")
    
    # Visualize timeline
    print("\nTimeline Visualization:")
    print(timeline_mgr.visualize_timeline(timeline))
    
    # Save to file
    example_dir = file_mgr.get_abs_path("examples/output")
    file_mgr.ensure_dir_exists(example_dir)
    output_path = example_dir / "manipulated_timeline.json"
    timeline_mgr.serialize_timeline(timeline, output_path)
    print(f"Timeline saved to {output_path}")

def timeline_serialization_example():
    """Example of advanced timeline serialization and deserialization"""
    print("\n=== Timeline Serialization & Deserialization ===")
    
    # Create a simple v3 timeline
    timeline = timeline_mgr.create_v3_timeline(
        width=1920,
        height=1080,
        framerate=29.97,
        samplerate=48000
    )
    
    # Add metadata during serialization
    example_dir = file_mgr.get_abs_path("examples/output")
    file_mgr.ensure_dir_exists(example_dir)
    output_path = example_dir / "serialized_timeline.json"
    
    # Serialize with description and custom name
    print("Serializing timeline with metadata...")
    timeline_mgr.serialize_timeline(
        timeline, 
        output_path,
        description="Demo timeline with metadata for project X"
    )
    
    # Deserialize the timeline
    print(f"Deserializing timeline from {output_path}...")
    loaded_timeline = timeline_mgr.deserialize_timeline(output_path)
    
    if loaded_timeline:
        print("Successfully deserialized timeline")
        print(f"Resolution: {loaded_timeline.res[0]}x{loaded_timeline.res[1]}")
        print(f"Timebase: {loaded_timeline.tb}")
    else:
        print("Failed to deserialize timeline")
    
    # Using convenience methods
    print("\nUsing convenience methods:")
    timeline_mgr.save_timeline(
        timeline, 
        "convenience_timeline",
        description="Saved using convenience method"
    )
    
    timeline_path = timeline_mgr.get_timeline_path("convenience_timeline")
    print(f"Timeline saved to: {timeline_path}")
    
    loaded_via_convenience = timeline_mgr.load_timeline("convenience_timeline")
    if loaded_via_convenience:
        print("Successfully loaded timeline using convenience method")
    
def timeline_format_conversion_example():
    """Example of timeline format conversion"""
    print("\n=== Timeline Format Conversion ===")
    
    # Check if we have a sample video
    sample_path = file_mgr.get_abs_path("clips/sample_clips/placeholder.mp4")
    if not file_mgr.file_exists(sample_path):
        print(f"Sample video not found at {sample_path}")
        print(f"This example requires a sample video at: {sample_path}")
        return
    
    # Create a v1 timeline
    print("Creating v1 timeline...")
    v1_timeline = timeline_mgr.create_v1_timeline(sample_path)
    
    # Print basic info
    print(f"Created v1 timeline with source: {v1_timeline.source.path.name}")
    
    # Convert to v3 timeline
    print("Converting v1 timeline to v3...")
    v3_timeline = timeline_mgr.convert_v1_to_v3(v1_timeline)
    
    # Print details
    if v3_timeline:
        print("Successfully converted to v3 timeline")
        print(f"Resolution: {v3_timeline.res[0]}x{v3_timeline.res[1]}")
        video_track_clips = len(v3_timeline.v[0]) if len(v3_timeline.v) > 0 else 0
        print(f"Video clips: {video_track_clips}")
        
        # Try to convert back to v1
        print("\nConverting v3 timeline back to v1...")
        v1_again = timeline_mgr.convert_v3_to_v1(v3_timeline)
        if v1_again:
            print("Successfully converted back to v1 format")
        else:
            print("Could not convert back to v1 format")
    else:
        print("Failed to convert to v3 timeline")
    
    # Serialize both formats
    example_dir = file_mgr.get_abs_path("examples/output")
    file_mgr.ensure_dir_exists(example_dir)
    
    # Save v1 format
    v1_path = example_dir / "v1_timeline.json"
    timeline_mgr.serialize_timeline(v1_timeline, v1_path)
    print(f"Saved v1 timeline to {v1_path}")
    
    # Save v3 format if available
    if v3_timeline:
        v3_path = example_dir / "v3_from_v1_timeline.json"
        timeline_mgr.serialize_timeline(v3_timeline, v3_path)
        print(f"Saved v3 timeline to {v3_path}")

def timeline_visualization_example():
    """Example demonstrating the enhanced timeline visualization capabilities"""
    print("\n=== Timeline Visualization Capabilities ===")
    
    # Check if we have a sample video
    sample_path = file_mgr.get_abs_path("clips/sample_clips/placeholder.mp4")
    if not file_mgr.file_exists(sample_path):
        print(f"Sample video not found at {sample_path}")
        print(f"This example requires a sample video at: {sample_path}")
        return
    
    # Create a v1 timeline
    v1_timeline = timeline_mgr.create_v1_timeline(sample_path)
    
    # Create a more complex v3 timeline
    v3_timeline = timeline_mgr.create_v3_timeline(
        source_path=sample_path,
        width=1920,
        height=1080,
        framerate=30
    )
    
    # Add some clips to the v3 timeline
    from auto_editor.timeline import TlVideo, TlAudio, TlRect
    from auto_editor.ffwrapper import initFileInfo
    
    # Initialize the source
    src = initFileInfo(str(sample_path), timeline_mgr.log)
    
    # Add video clips to the timeline
    v3_timeline.v[0].append(TlVideo(start=0, dur=90, src=src, offset=0, speed=1.0, stream=0))
    v3_timeline.v[0].append(TlVideo(start=90, dur=120, src=src, offset=30, speed=1.5, stream=0))
    
    # Add a second video track with a rectangle
    if len(v3_timeline.v) == 1:
        v3_timeline.v.append([])
    v3_timeline.v[1].append(TlRect(start=30, dur=60, x=100, y=100, width=400, height=200, fill="#FF0000"))
    
    # Add audio if available
    if hasattr(src, 'audio') and src.audio:
        v3_timeline.a[0].append(TlAudio(start=0, dur=90, src=src, offset=0, speed=1.0, volume=1.0, stream=0))
    
    # Demonstrate different detail levels for v1 timeline
    print("\nV1 Timeline - Minimal Detail:")
    v1_minimal = timeline_mgr.visualize_timeline(v1_timeline, detail_level='minimal')
    print(v1_minimal)
    
    print("\nV1 Timeline - Normal Detail:")
    v1_normal = timeline_mgr.visualize_timeline(v1_timeline, detail_level='normal')
    print(v1_normal)
    
    print("\nV1 Timeline - Detailed:")
    v1_detailed = timeline_mgr.visualize_timeline(v1_timeline, detail_level='detailed')
    print(v1_detailed)
    
    # Demonstrate different detail levels for v3 timeline
    print("\nV3 Timeline - Summary:")
    timeline_mgr.print_timeline_summary(v3_timeline)
    
    print("\nV3 Timeline - Normal Detail:")
    v3_normal = timeline_mgr.visualize_timeline(v3_timeline, detail_level='normal')
    print(v3_normal)
    
    # Export visualizations to files
    example_dir = file_mgr.get_abs_path("examples/output")
    v1_viz_path = example_dir / "v1_timeline_visualization.txt"
    v3_viz_path = example_dir / "v3_timeline_visualization.txt"
    
    timeline_mgr.export_timeline_visualization(v1_timeline, v1_viz_path, detail_level='detailed')
    timeline_mgr.export_timeline_visualization(v3_timeline, v3_viz_path, detail_level='detailed')
    
    print(f"\nExported timeline visualizations to:")
    print(f"  - {v1_viz_path}")
    print(f"  - {v3_viz_path}")
    
    # Demonstrate custom width visualization
    print("\nWide visualization (120 chars):")
    wide_viz = timeline_mgr.visualize_timeline(v3_timeline, width=120, detail_level='minimal')
    print(wide_viz)

if __name__ == "__main__":
    print("TimelineManager Examples")
    print("=======================")
    
    # Run examples
    basic_timeline_example()
    clip_sequence_example()
    timeline_manipulation_example()
    timeline_serialization_example()
    timeline_format_conversion_example()
    timeline_visualization_example()
    
    print("\nExamples completed. Check the examples/output directory for timeline files and visualizations.")