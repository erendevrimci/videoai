#!/usr/bin/env python3
"""
Main entry point for the VideoAI automation pipeline.

This script orchestrates the entire video creation process:
1. Script generation
2. Voice-over generation
3. Caption generation
4. Video editing and assembly
5. Title and description generation
6. YouTube upload
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

# Initialize logging system
try:
    from logging_system import Logger, LogLevel, logger, setup_thread_exception_handling
    # Set up thread exception handling
    setup_thread_exception_handling()
    has_logging = True
except ImportError:
    has_logging = False
    
# Import pipeline modules
import write_script
import voice_over
import video_edit
import write_title_desc
import upload_video
from captions import generate_subtitles
from config import config, get_channel_config, get_timeline_config
from file_manager import FileManager
from timeline_manager import TimelineManager
from auto_editor.timeline import v3

# Initialize the file manager
file_mgr = FileManager()

# Create a module-specific logger
if has_logging:
    logger = Logger.get_logger("main")

def process_channel(channel_number: int, steps: List[str] = None) -> None:
    """
    Process a single channel through the complete video pipeline.
    
    This is the legacy processing function without timeline integration.
    For timeline-aware processing, see process_channel_with_timeline.
    
    Args:
        channel_number (int): The channel number to process
        steps (List[str], optional): List of specific steps to run. If None, runs all steps.
    """
    if steps is None:
        steps = ["script", "voice", "captions", "video", "title", "upload"]
    
    if has_logging:
        logger.info(f"Starting process for Channel {channel_number}")
        logger.info(f"Running steps: {', '.join(steps)}")
    else:
        print(f"\nStarting process for Channel {channel_number}...\n")
        print(f"Running steps: {', '.join(steps)}")
    
    try:
        channel_config = get_channel_config(channel_number)
        if has_logging:
            logger.info(f"Using voice ID: {channel_config.voice_id}")
        else:
            print(f"Using voice ID: {channel_config.voice_id}")
        
        # Create channel output directory using file manager
        channel_dir = file_mgr.get_channel_output_path(channel_number)
        if has_logging:
            logger.info(f"Output directory: {channel_dir}")
        else:
            print(f"Output directory: {channel_dir}")
        
        # 1. Generate the YouTube script
        if "script" in steps:
            step_msg = "--- Step 1: Generating Script ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            write_script.main(channel_number)
            
            # Verify that the script was generated (this helps with debugging)
            file_paths_json_path = file_mgr.get_channel_output_path(channel_number) / "current_file_paths.json"
            dynamic_file_paths = file_mgr.read_json(file_paths_json_path)
            if dynamic_file_paths and "script_file" in dynamic_file_paths:
                script_file = file_mgr.get_channel_output_path(channel_number) / dynamic_file_paths["script_file"]
                if has_logging:
                    logger.info(f"Script generated at: {script_file}")
                else:
                    print(f"Script generated at: {script_file}")
        
        # 2. Generate the voice-over using ElevenLabs
        if "voice" in steps:
            step_msg = "--- Step 2: Generating Voice-over ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            voice_over.main(channel_number)
        
        # 3. Generate captions (SRT file)
        if "captions" in steps:
            step_msg = "--- Step 3: Generating Captions ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Check for dynamic file paths
            file_paths_json_path = file_mgr.get_channel_output_path(channel_number) / "current_file_paths.json"
            dynamic_file_paths = file_mgr.read_json(file_paths_json_path)
            
            # Use channel-specific paths for voice and captions (using dynamic paths if available)
            if dynamic_file_paths and "voice_file" in dynamic_file_paths:
                voice_file_name = dynamic_file_paths["voice_file"].replace("voice/", "")
                voice_file = file_mgr.get_audio_output_path(channel_number, voice_file_name.replace(".mp3", ""))
                print(f"Using dynamic voice file for captions: {voice_file}")
            else:
                voice_file = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/",""))
                
            if dynamic_file_paths and "captions_file" in dynamic_file_paths:
                captions_file = file_mgr.get_channel_output_path(channel_number) / dynamic_file_paths["captions_file"]
                print(f"Using dynamic captions file path: {captions_file}")
            else:
                captions_file = file_mgr.get_caption_path(channel_number, config.file_paths.captions_file)
            
            print(f"Using voice file: {voice_file}")
            print(f"Output captions to: {captions_file}")
            
            # Ensure the audio dir exists
            file_mgr.ensure_dir_exists(voice_file.parent)
            
            generate_subtitles(str(voice_file), str(captions_file), channel_number)
        
        # 4. Edit and assemble the video
        if "video" in steps:
            step_msg = "--- Step 4: Editing Video ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            video_edit.main(channel_number)
        
        # 5. Generate the title and description
        if "title" in steps:
            step_msg = "--- Step 5: Generating Title & Description ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            write_title_desc.main(channel_number)
        
        # 6. Upload the video to YouTube
        if "upload" in steps:
            step_msg = f"--- Step 6: Uploading Video to YouTube Channel {channel_number} ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            upload_video.main(channel_number)
            
        # Archive files for this channel
        if "archive" in steps:
            archive_channel_files(channel_number)
        
        success_msg = f"Channel {channel_number} processing completed successfully."
        if has_logging:
            logger.info(success_msg)
        else:
            print(f"\n{success_msg}")
    
    except Exception as e:
        error_msg = f"Error processing channel {channel_number}: {str(e)}"
        if has_logging:
            logger.error(error_msg, exc_info=True)
        else:
            print(f"\n{error_msg}")
            # Print the full traceback for better debugging
            import traceback
            traceback.print_exc()

def archive_channel_files(channel_number: int) -> None:
    """
    Archive important files for a channel by copying them to the channel output directory.
    
    Args:
        channel_number (int): The channel number
    """
    channel_dir = file_mgr.get_channel_output_path(channel_number)
    
    # Define source and target files
    files_to_archive = {
        
        "youtube_info.json": f"youtube_info_channel{channel_number}.json"
    }
    
    archive_msg = f"Archiving files for channel {channel_number}..."
    if has_logging:
        logger.info(archive_msg)
    else:
        print(f"\n{archive_msg}")
    
    # Archive each file using file manager
    for source, target in files_to_archive.items():
        source_path = file_mgr.get_abs_path(source)
        target_path = channel_dir / target
        
        # Use the file manager to copy the files
        if file_mgr.file_exists(source_path):
            # Copy file directly using FileManager
            success = file_mgr.copy_file(source_path, target_path)
            if success:
                success_msg = f"Archived {source} to {target_path}"
                if has_logging:
                    logger.info(success_msg)
                else:
                    print(f"  {success_msg}")
            else:
                error_msg = f"Could not archive {source} to {target_path}"
                if has_logging:
                    logger.warning(error_msg)
                else:
                    print(f"  Warning: {error_msg}")
        else:
            warning_msg = f"Source file {source_path} not found"
            if has_logging:
                logger.warning(warning_msg)
            else:
                print(f"  Warning: {warning_msg}")

def parse_srt_for_timeline(srt_path: Path) -> List[Dict[str, Any]]:
    """
    Parse an SRT file to extract timing information for timeline construction.
    
    Args:
        srt_path: Path to the SRT caption file
        
    Returns:
        List of dictionaries with start_time, end_time, duration, and text fields
    """
    if not file_mgr.file_exists(srt_path):
        if has_logging:
            logger.warning(f"SRT file not found: {srt_path}")
        else:
            print(f"Warning: SRT file not found: {srt_path}")
        return []
    
    srt_content = file_mgr.read_text(srt_path)
    if not srt_content:
        if has_logging:
            logger.warning(f"SRT file is empty: {srt_path}")
        else:
            print(f"Warning: SRT file is empty: {srt_path}")
        return []
    
    # Parse SRT segments
    segments = []
    srt_parts = [part.strip() for part in srt_content.split('\n\n') if part.strip()]
    
    for part in srt_parts:
        lines = part.split('\n')
        if len(lines) < 3:  # Need at least number, timestamp, and text
            continue
        
        # Extract timing information
        timestamp_line = lines[1]
        time_parts = timestamp_line.split(' --> ')
        if len(time_parts) != 2:
            continue
            
        # Convert SRT timestamps (HH:MM:SS,mmm) to seconds
        start_str, end_str = time_parts
        
        # Parse start time
        start_parts = start_str.replace(',', '.').split(':')
        if len(start_parts) != 3:
            continue
        start_time = (
            float(start_parts[0]) * 3600 +  # Hours
            float(start_parts[1]) * 60 +    # Minutes
            float(start_parts[2])           # Seconds
        )
        
        # Parse end time
        end_parts = end_str.replace(',', '.').split(':')
        if len(end_parts) != 3:
            continue
        end_time = (
            float(end_parts[0]) * 3600 +    # Hours
            float(end_parts[1]) * 60 +      # Minutes
            float(end_parts[2])             # Seconds
        )
        
        # Calculate duration
        duration = end_time - start_time
        
        # Extract text (may be multiple lines)
        text = ' '.join(lines[2:])
        
        # Add segment data
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'text': text,
            'index': len(segments)
        })
    
    if has_logging:
        logger.info(f"Extracted {len(segments)} caption segments from SRT file")
    else:
        print(f"Extracted {len(segments)} caption segments from SRT file")
    
    return segments

def add_script_segments_to_timeline(timeline: v3, script_segments: List[Dict[str, Any]]) -> v3:
    """
    Add script segments to a timeline as metadata and markers.
    
    Args:
        timeline: The timeline to add script segments to
        script_segments: List of script segments with text, start_time, and end_time keys
        
    Returns:
        Updated timeline with script segments added
    """
    if has_logging:
        logger.info(f"Adding {len(script_segments)} script segments to timeline")
    else:
        print(f"Adding {len(script_segments)} script segments to timeline")
    
    # Add script segments as timeline metadata
    if not hasattr(timeline, 'metadata'):
        timeline.metadata = {}
        
    # Add full script text
    full_script = "\n".join([segment.get('text', '') for segment in script_segments])
    timeline.metadata['script'] = full_script
    
    # Add segmented script data
    timeline.metadata['script_segments'] = script_segments
    
    # Store script length stats
    word_count = len(full_script.split())
    timeline.metadata['script_word_count'] = word_count
    timeline.metadata['script_segment_count'] = len(script_segments)
    
    # In a future implementation, we could add visual markers or text objects
    # to the timeline for each script segment, but this requires extending
    # the timeline model with marker support
    
    return timeline

def dummy_log():
    """Create a dummy log object for auto_editor components"""
    class DummyLog:
        def __init__(self):
            pass
        def print(self, *args, **kwargs):
            pass
        def debug(self, *args, **kwargs):
            pass
        def error(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
    return DummyLog()

def add_voice_to_timeline(timeline: v3, voice_file: Path, script_segments: List[Dict[str, Any]] = None) -> v3:
    """
    Add voice-over audio to a timeline.
    
    Args:
        timeline: The timeline to add voice-over to
        voice_file: Path to the voice-over audio file
        script_segments: Optional list of script segments with timing information
        
    Returns:
        Updated timeline with voice-over added
    """
    if has_logging:
        logger.info(f"Adding voice-over from {voice_file} to timeline")
    else:
        print(f"Adding voice-over from {voice_file} to timeline")
    
    if not file_mgr.file_exists(voice_file):
        if has_logging:
            logger.warning(f"Voice file not found: {voice_file}")
        else:
            print(f"Warning: Voice file not found: {voice_file}")
        return timeline
    
    try:
        # Import ffwrapper components
        from auto_editor.ffwrapper import initFileInfo
        from auto_editor.timeline import TlAudio
        
        # Initialize the voice source
        voice_src = initFileInfo(str(voice_file), dummy_log()) 
        
        # Ensure we have at least one audio track
        if not timeline.a or len(timeline.a) == 0:
            timeline.a = [[]]
        
        # If we have script segments with timing, create multiple audio segments
        # aligned with the script timing
        if script_segments and len(script_segments) > 0:
            for i, segment in enumerate(script_segments):
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                duration = segment.get('duration', 0)
                
                if duration <= 0:
                    continue
                
                # Convert to frames for timeline
                start_frame = int(start_time * float(timeline.tb))
                duration_frames = int(duration * float(timeline.tb))
                
                # Get correct audio samplerate from the voice source
                # Voice source is a FileInfo object with audios attribute
                if voice_src.audios and len(voice_src.audios) > 0:
                    # Use the samplerate from the first audio stream
                    offset_frames = int(start_time * voice_src.audios[0].samplerate)
                else:
                    # Fallback to a default if we can't get samplerate
                    offset_frames = int(start_time * 48000)  # Default to 48kHz
                
                # Create audio object for this segment
                audio_obj = TlAudio(
                    start=start_frame,
                    dur=duration_frames,
                    src=voice_src,
                    offset=offset_frames,
                    speed=1.0,
                    volume=1.0,  # Will be adjusted in the video processing step
                    stream=0
                )
                
                # Add to the first audio track
                timeline.a[0].append(audio_obj)
                
            if has_logging:
                logger.info(f"Added {len(script_segments)} voice segments to timeline")
            else:
                print(f"Added {len(script_segments)} voice segments to timeline")
        else:
            # Add the entire voice file as a single audio segment
            # Get audio duration from the first audio stream
            audio_duration = 0
            if voice_src.audios and len(voice_src.audios) > 0:
                audio_duration = voice_src.audios[0].duration
                
            # Convert to frames for timeline
            duration_frames = int(audio_duration * float(timeline.tb))
            
            # If we couldn't get the duration, use a reasonable default
            if duration_frames <= 0:
                duration_frames = int(60 * float(timeline.tb))  # 60 seconds
            
            # Create audio object
            audio_obj = TlAudio(
                start=0,
                dur=duration_frames,
                src=voice_src,
                offset=0,
                speed=1.0,
                volume=1.0,  # Will be adjusted in the video processing step
                stream=0
            )
            
            # Add to the first audio track
            timeline.a[0].append(audio_obj)
            
            if has_logging:
                logger.info(f"Added complete voice-over ({audio_duration:.2f}s) to timeline")
            else:
                print(f"Added complete voice-over ({audio_duration:.2f}s) to timeline")
        
        # Add voice-over metadata
        if not hasattr(timeline, 'metadata'):
            timeline.metadata = {}
        timeline.metadata['voice_file'] = str(voice_file)
        
        return timeline
        
    except Exception as e:
        if has_logging:
            logger.error(f"Error adding voice to timeline: {e}", exc_info=True)
        else:
            print(f"Error adding voice to timeline: {e}")
            traceback.print_exc()
        
        return timeline

def add_clips_to_timeline(timeline: v3, clip_sequence: List[Dict[str, Any]]) -> v3:
    """
    Add video clips to a timeline.
    
    Args:
        timeline: The timeline to add clips to
        clip_sequence: List of clip dictionaries with clip_name, start_time, duration, etc.
        
    Returns:
        Updated timeline with clips added
    """
    if has_logging:
        logger.info(f"Adding {len(clip_sequence)} clips to timeline")
    else:
        print(f"Adding {len(clip_sequence)} clips to timeline")
    
    # Get a TimelineManager instance for creating timeline objects
    timeline_mgr = TimelineManager()
    
    # Get the base clips directory
    clips_dir = file_mgr.get_abs_path(config.file_paths.clips_directory)
    
    # Ensure we have at least one video track
    if not timeline.v or len(timeline.v) == 0:
        timeline.v = [[]]
    
    try:
        # Import required components
        from auto_editor.ffwrapper import initFileInfo
        from auto_editor.timeline import TlVideo
        
        # Current position in timeline (in frames)
        current_frame = 0
        
        # Process each clip in the sequence
        for i, clip in enumerate(clip_sequence):
            clip_name = clip.get('clip_name', '')
            clip_start = clip.get('start_time', 0)  # Start time within the clip
            clip_duration = clip.get('duration', 5.0)  # Duration to use from the clip
            
            # Find the clip file
            # Look in different possible locations
            possible_paths = [
                clips_dir / clip_name,
                clips_dir / f"{clip_name}.mp4",
                clips_dir / "sample_clips" / clip_name,
                clips_dir / "sample_clips" / f"{clip_name}.mp4"
            ]
            
            clip_path = None
            for path in possible_paths:
                if file_mgr.file_exists(path):
                    clip_path = path
                    break
            
            # If clip not found, use a placeholder
            placeholder_used = False
            if not clip_path:
                if has_logging:
                    logger.warning(f"Clip not found: {clip_name}. Using placeholder.")
                else:
                    print(f"Warning: Clip not found: {clip_name}. Using placeholder.")
                
                placeholder_path = clips_dir / "sample_clips" / "placeholder.mp4"
                if file_mgr.file_exists(placeholder_path):
                    clip_path = placeholder_path
                    placeholder_used = True
                else:
                    if has_logging:
                        logger.error(f"Placeholder clip not found at {placeholder_path}")
                    else:
                        print(f"Error: Placeholder clip not found at {placeholder_path}")
                    continue
            
            # Initialize the clip source
            try:
                clip_src = initFileInfo(str(clip_path), dummy_log())
            except Exception as e:
                if has_logging:
                    logger.error(f"Error initializing clip source: {e}")
                else:
                    print(f"Error initializing clip source: {e}")
                continue
            
            # Calculate frame numbers
            clip_start_frame = int(clip_start * float(timeline.tb))  # Start frame in source
            clip_duration_frames = int(clip_duration * float(timeline.tb))  # Duration in frames
            
            # Create video object
            video_obj = TlVideo(
                start=current_frame,  # Position in timeline
                dur=clip_duration_frames,  # Duration to use
                src=clip_src,  # Source file
                offset=clip_start_frame,  # Start position in source
                speed=1.0,  # Normal speed
                stream=0  # Main video stream
            )
            
            # Add to the first video track
            timeline.v[0].append(video_obj)
            
            # Add clip metadata if needed
            if 'script_segment' in clip and clip['script_segment']:
                # Store the script segment for this clip if available
                video_obj.script_segment = clip['script_segment']
            
            # Update current position
            current_frame += clip_duration_frames
            
            if has_logging:
                logger.info(f"Added clip {i+1}: {clip_name} at position {current_frame - clip_duration_frames}")
            else:
                print(f"Added clip {i+1}: {clip_name} at position {current_frame - clip_duration_frames}")
        
        # Store the clip sequence in timeline metadata
        if not hasattr(timeline, 'metadata'):
            timeline.metadata = {}
        timeline.metadata['clip_sequence'] = clip_sequence
        
        # Calculate total duration of the timeline
        total_frames = sum(clip.get('duration', 0) * float(timeline.tb) for clip in clip_sequence)
        total_seconds = total_frames / float(timeline.tb)
        if has_logging:
            logger.info(f"Total timeline duration: {total_seconds:.2f} seconds ({int(total_frames)} frames)")
        else:
            print(f"Total timeline duration: {total_seconds:.2f} seconds ({int(total_frames)} frames)")
        
        return timeline
    
    except Exception as e:
        if has_logging:
            logger.error(f"Error adding clips to timeline: {e}", exc_info=True)
        else:
            print(f"Error adding clips to timeline: {e}")
            traceback.print_exc()
        
        # Return the original timeline even if we had an error
        return timeline

def process_channel_with_timeline(channel_number: int, steps: List[str] = None) -> bool:
    """
    Process a single channel through the pipeline with timeline integration.
    
    This version of the processing function creates a timeline early in the
    pipeline and progressively builds it throughout each step.
    
    Args:
        channel_number: The channel number to process
        steps: List of specific steps to run. If None, runs all steps.
        
    Returns:
        bool: True if successful, False otherwise
    """
    if steps is None:
        steps = ["script", "voice", "captions", "video", "title", "upload"]
    
    step_msg = f"Starting timeline-aware processing for Channel {channel_number}"
    if has_logging:
        logger.info(step_msg)
        logger.info(f"Running steps: {', '.join(steps)}")
    else:
        print(f"\n{step_msg}")
        print(f"Running steps: {', '.join(steps)}")
    
    try:
        # Get channel configuration
        channel_config = get_channel_config(channel_number)
        timeline_config = get_timeline_config(channel_number)
        
        # Create channel output directory
        channel_dir = file_mgr.get_channel_output_path(channel_number)
        if has_logging:
            logger.info(f"Output directory: {channel_dir}")
        else:
            print(f"Output directory: {channel_dir}")
        
        # Initialize TimelineManager for this channel
        timeline_mgr = TimelineManager(channel_number=channel_number)
        
        # Create initial empty timeline with channel-specific settings
        timeline = timeline_mgr.create_v3_timeline(
            width=timeline_config.default_width,
            height=timeline_config.default_height,
            framerate=timeline_config.default_framerate
        )
        
        # Initialize metadata for the timeline
        if not hasattr(timeline, 'metadata'):
            timeline.metadata = {}
        
        # Add channel info to metadata
        timeline.metadata['channel'] = channel_number
        timeline.metadata['timestamp'] = time.time()
        timeline.metadata['steps'] = steps
        
        # Save initial empty timeline
        timeline_path = file_mgr.get_timeline_path("initial_timeline", channel_number)
        timeline_mgr.serialize_timeline(timeline, timeline_path, "Initial empty timeline")
        
        # 1. Generate the YouTube script
        if "script" in steps:
            step_msg = "--- Step 1: Generating Script with Timeline Integration ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Generate script normally
            write_script.main(channel_number)
            
            # Check for dynamic file paths
            file_paths_json_path = file_mgr.get_channel_output_path(channel_number) / "current_file_paths.json"
            dynamic_file_paths = file_mgr.read_json(file_paths_json_path)
            
            # Read the generated script to extract segments
            if dynamic_file_paths and "script_file" in dynamic_file_paths:
                script_file = file_mgr.get_channel_output_path(channel_number) / dynamic_file_paths["script_file"]
                print(f"Using dynamic script file path: {script_file}")
            else:
                script_file = file_mgr.get_script_path(channel_number)
                
            script_content = file_mgr.read_text(script_file)
            
            if script_content:
                # Store script metadata before segmentation
                timeline.metadata['script_file'] = str(script_file)
                timeline.metadata['script_character_count'] = len(script_content)
                timeline.metadata['script_word_count'] = len(script_content.split())
                
                # In a full implementation, we would parse the script into detailed segments
                # For now, create a simple segment structure
                script_segments = [
                    {
                        "text": script_content,
                        "start_time": 0, 
                        "end_time": len(script_content.split()) / 2,
                        "index": 0,
                        "segment_type": "narrative"
                    }
                ]
                
                # Add script segments to timeline
                timeline = add_script_segments_to_timeline(timeline, script_segments)
                
                # Save updated timeline after script generation
                timeline_mgr.save_timeline(timeline, "script_timeline", "Timeline after script generation")
                
                if has_logging:
                    logger.info(f"Added script with {len(script_content.split())} words to timeline")
                else:
                    print(f"Added script with {len(script_content.split())} words to timeline")
        
        # 2. Generate the voice-over
        if "voice" in steps:
            step_msg = "--- Step 2: Generating Voice-over with Timeline Integration ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Generate voice normally
            voice_over.main(channel_number)
            
            # Check for dynamic file paths
            file_paths_json_path = file_mgr.get_channel_output_path(channel_number) / "current_file_paths.json"
            dynamic_file_paths = file_mgr.read_json(file_paths_json_path)
            
            # Get voice file path (using dynamic path if available)
            if dynamic_file_paths and "voice_file" in dynamic_file_paths:
                voice_file_name = dynamic_file_paths["voice_file"].replace("voice/", "")
                voice_file = file_mgr.get_audio_output_path(channel_number, voice_file_name.replace(".mp3", ""))
                print(f"Using dynamic voice file path: {voice_file}")
            else:
                voice_file = file_mgr.get_audio_output_path(
                    channel_number, 
                    config.file_paths.voice_file.replace("voice/","")
                )
            
            # Add voice to timeline
            if file_mgr.file_exists(voice_file):
                # Check if we have captions already to use for better timing
                captions_file = file_mgr.get_caption_path(channel_number, config.file_paths.captions_file)
                caption_segments = []
                
                # Try to read captions file to extract timing information
                if file_mgr.file_exists(captions_file):
                    caption_segments = parse_srt_for_timeline(captions_file)
                
                # Add voice to timeline using caption segments if available
                if caption_segments:
                    timeline = add_voice_to_timeline(timeline, voice_file, caption_segments)
                else:
                    timeline = add_voice_to_timeline(timeline, voice_file)
                
                # Calculate voice duration for metadata
                from auto_editor.ffwrapper import initFileInfo
                try:
                    voice_src = initFileInfo(str(voice_file), dummy_log())
                    if hasattr(voice_src, 'audio') and voice_src.audio:
                        timeline.metadata['voice_duration'] = voice_src.audio.duration
                except Exception as e:
                    if has_logging:
                        logger.warning(f"Could not determine voice duration: {e}")
                
                # Save updated timeline after voice generation
                timeline_mgr.save_timeline(timeline, "voice_timeline", "Timeline after voice generation")
                
                if has_logging:
                    logger.info(f"Added voice-over from {voice_file} to timeline")
                else:
                    print(f"Added voice-over from {voice_file} to timeline")
        
        # 3. Generate captions (SRT file)
        if "captions" in steps:
            step_msg = "--- Step 3: Generating Captions with Timeline Integration ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Check for dynamic file paths
            file_paths_json_path = file_mgr.get_channel_output_path(channel_number) / "current_file_paths.json"
            dynamic_file_paths = file_mgr.read_json(file_paths_json_path)
            
            # Get channel-specific paths (using dynamic paths if available)
            if dynamic_file_paths and "voice_file" in dynamic_file_paths:
                voice_file_name = dynamic_file_paths["voice_file"].replace("voice/", "")
                voice_file = file_mgr.get_audio_output_path(channel_number, voice_file_name.replace(".mp3", ""))
                print(f"Using dynamic voice file path for captions: {voice_file}")
            else:
                voice_file = file_mgr.get_audio_output_path(
                    channel_number, 
                    config.file_paths.voice_file.replace("voice/","")
                )
                
            if dynamic_file_paths and "captions_file" in dynamic_file_paths:
                captions_file = file_mgr.get_channel_output_path(channel_number) / dynamic_file_paths["captions_file"]
                print(f"Using dynamic captions file path: {captions_file}")
            else:
                captions_file = file_mgr.get_caption_path(channel_number, config.file_paths.captions_file)
            
            # Ensure directory exists
            file_mgr.ensure_dir_exists(voice_file.parent)
            
            # Generate subtitles normally
            generate_subtitles(str(voice_file), str(captions_file), channel_number)
            
            # Parse the captions to get precise word timings
            if file_mgr.file_exists(captions_file):
                caption_segments = parse_srt_for_timeline(captions_file)
                
                if caption_segments:
                    # Store caption segments in timeline metadata
                    timeline.metadata['caption_segments'] = caption_segments
                    timeline.metadata['caption_count'] = len(caption_segments)
                    
                    # Calculate total caption duration
                    if caption_segments:
                        total_duration = sum(segment.get('duration', 0) for segment in caption_segments)
                        timeline.metadata['caption_duration'] = total_duration
                    
                    # If we have voice audio track, update it with more precise timing
                    if timeline.a and len(timeline.a) > 0 and len(timeline.a[0]) > 0:
                        # In a more advanced implementation, we would:
                        # 1. Update voice timing based on caption segments
                        # 2. Create separate audio segments for each caption
                        # 3. Synchronize script segments with voice segments
                        
                        # For now, just update metadata with caption info
                        timeline.metadata['caption_file'] = str(captions_file)
                    
                    if has_logging:
                        logger.info(f"Added {len(caption_segments)} caption segments to timeline")
                    else:
                        print(f"Added {len(caption_segments)} caption segments to timeline")
            
            # Save updated timeline after caption generation
            timeline_mgr.save_timeline(timeline, "captions_timeline", "Timeline after caption generation")
        
        # 4. Edit and assemble the video
        if "video" in steps:
            step_msg = "--- Step 4: Editing Video with Timeline Integration ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Run the video edit process with the timeline
            # The timeline will be updated with video clips and potentially rendered
            result = video_edit.main(channel_number, timeline_mode=True, timeline=timeline)
            
            # The timeline should now be updated with clips
            # Update timing information for the final timeline
            try:
                if hasattr(timeline, 'end'):
                    frames = timeline.end
                    fps = float(timeline.tb)
                    duration_sec = frames / fps
                    
                    # Update timing metadata
                    timeline.metadata['total_frames'] = frames
                    timeline.metadata['total_duration'] = duration_sec
                    
                    if has_logging:
                        logger.info(f"Final timeline duration: {duration_sec:.2f} seconds ({frames} frames)")
                    else:
                        print(f"Final timeline duration: {duration_sec:.2f} seconds ({frames} frames)")
            except Exception as e:
                if has_logging:
                    logger.warning(f"Could not update timing metadata: {e}")
                else:
                    print(f"Warning: Could not update timing metadata: {e}")
            
            # Save the final timeline
            timeline_mgr.save_timeline(timeline, "final_timeline", "Final video timeline")
            
            # Create a visualization of the timeline
            try:
                visualization_path = timeline_mgr.export_timeline_visualization(
                    timeline, 
                    detail_level="detailed"
                )
                
                if has_logging:
                    logger.info(f"Timeline visualization exported to: {visualization_path}")
                else:
                    print(f"Timeline visualization exported to: {visualization_path}")
            except Exception as e:
                if has_logging:
                    logger.warning(f"Could not export timeline visualization: {e}")
                else:
                    print(f"Warning: Could not export timeline visualization: {e}")
        
        # 5. Generate the title and description
        if "title" in steps:
            step_msg = "--- Step 5: Generating Title & Description ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Generate title and description normally
            write_title_desc.main(channel_number)
            
            # Get the generated title and description
            title_desc_file = file_mgr.get_title_desc_path(channel_number)
            if file_mgr.file_exists(title_desc_file):
                title_desc_content = file_mgr.read_json(title_desc_file)
                
                if title_desc_content:
                    # Add to timeline metadata
                    timeline.metadata['video_title'] = title_desc_content.get('title', '')
                    timeline.metadata['video_description'] = title_desc_content.get('description', '')
                    
                    # Save updated timeline after title/description generation
                    timeline_mgr.save_timeline(timeline, "metadata_timeline", "Timeline with video metadata")
                    
                    if has_logging:
                        logger.info(f"Added title and description to timeline metadata")
                    else:
                        print(f"Added title and description to timeline metadata")
        
        # 6. Upload the video to YouTube
        if "upload" in steps:
            step_msg = f"--- Step 6: Uploading Video to YouTube Channel {channel_number} ---"
            if has_logging:
                logger.info(step_msg)
            else:
                print(f"\n{step_msg}")
            
            # Upload video normally
            upload_result = upload_video.main(channel_number)
            
            # Add upload result to timeline metadata
            timeline.metadata['upload_completed'] = upload_result
            
            if upload_result:
                # Get YouTube info for video ID using the channel-specific path
                youtube_info_file = file_mgr.get_channel_output_path(channel_number) / f"youtube_info_channel{channel_number}.json"
                if file_mgr.file_exists(youtube_info_file):
                    youtube_info = file_mgr.read_json(youtube_info_file)
                    
                    if youtube_info and 'video_id' in youtube_info:
                        timeline.metadata['youtube_video_id'] = youtube_info['video_id']
                        
                        if has_logging:
                            logger.info(f"Added YouTube video ID to timeline metadata: {youtube_info['video_id']}")
                        else:
                            print(f"Added YouTube video ID to timeline metadata: {youtube_info['video_id']}")
            
            # Save final timeline with upload information
            timeline_mgr.save_timeline(timeline, "published_timeline", "Timeline for published video")
            
        # Archive files for this channel
        if "archive" in steps:
            archive_channel_files(channel_number)
            
            # Also archive the timeline
            try:
                # Create a timestamped archive name
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"archived_timeline_{timestamp}"
                
                # Save archived timeline
                timeline_mgr.save_timeline(timeline, archive_name, f"Archived timeline from {datetime.now().isoformat()}")
                
                if has_logging:
                    logger.info(f"Timeline archived as {archive_name}")
                else:
                    print(f"Timeline archived as {archive_name}")
            except Exception as e:
                if has_logging:
                    logger.warning(f"Could not archive timeline: {e}")
                else:
                    print(f"Warning: Could not archive timeline: {e}")
        
        success_msg = f"Channel {channel_number} timeline-aware processing completed successfully."
        if has_logging:
            logger.info(success_msg)
        else:
            print(f"\n{success_msg}")
            
        return True
    
    except Exception as e:
        error_msg = f"Error in timeline-aware processing for channel {channel_number}: {str(e)}"
        if has_logging:
            logger.error(error_msg, exc_info=True)
        else:
            print(f"\n{error_msg}")
            # Print the full traceback for better debugging
            import traceback
            traceback.print_exc()
            
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VideoAI Automation Pipeline")
    
    parser.add_argument(
        "--channel", 
        type=int, 
        choices=[1, 2, 3], 
        default=1,  # MODIFIED: Default to channel 1 only
        help="Channel number to process (1-3). Default: 1"
    )
    
    parser.add_argument(
        "--steps", 
        type=str, 
        default="script,voice,captions,video,title",  # MODIFIED: Removed upload step
        help=(
            "Comma-separated list of steps to run: "
            "script,voice,captions,video,title,upload,archive. Default: script,voice,captions,video,title"
        )
    )
    
    parser.add_argument(
        "--delay",
        type=int,
        default=60,
        help="Delay in seconds between processing channels. Default: 60"
    )
    
    # Timeline processing options
    timeline_group = parser.add_argument_group('Timeline Processing')
    
    timeline_exclusive = timeline_group.add_mutually_exclusive_group()
    timeline_exclusive.add_argument(
        "--timeline",
        action="store_true",
        help="Use timeline-aware processing pipeline (legacy option, timeline mode is now the default)"
    )
    timeline_exclusive.add_argument(
        "--no-timeline",
        action="store_true",
        help="Disable timeline-aware processing and use traditional pipeline"
    )
    
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save timeline checkpoints after each processing step"
    )
    
    args = parser.parse_args()
    print("\n⚠️ MODIFIED DEFAULTS: Processing only channel 1, YouTube upload disabled")
    return args

def main() -> None:
    """Main entry point for the VideoAI pipeline."""
    args = parse_arguments()
    
    # Process all steps or specific ones
    steps = args.steps.lower().split(",") if args.steps.lower() != "all" else None
    
    # Configure logging if available
    if has_logging:
        # Log the arguments
        logger.info(f"Arguments: channel={args.channel}, steps={args.steps}, delay={args.delay}, timeline={args.timeline}")
    
    # Determine whether to use timeline-aware processing
    # Default to timeline processing unless explicitly disabled with --no-timeline
    use_timeline = not args.no_timeline
    
    if use_timeline:
        # Check if timeline rendering is enabled in configuration
        timeline_config = get_timeline_config(args.channel)
        if not timeline_config.rendering.enabled:
            warning_msg = "Timeline processing requested but timeline rendering is disabled in config."
            warning_msg += " Enabling it for this run."
            if has_logging:
                logger.warning(warning_msg)
            else:
                print(f"\n⚠️ {warning_msg}")
            # Enable timeline rendering for this run
            timeline_config.rendering.enabled = True
        
        info_msg = "Using timeline-aware processing pipeline"
        if has_logging:
            logger.info(info_msg)
        else:
            print(f"\n{info_msg}")
    else:
        info_msg = "Using traditional processing pipeline (timeline mode disabled)"
        if has_logging:
            logger.info(info_msg)
        else:
            print(f"\n{info_msg}")
    
    # Process specific channel or all channels
    if args.channel:
        if has_logging:
            logger.info(f"Processing channel {args.channel}")
        
        if use_timeline:
            # Use timeline-aware processing
            process_channel_with_timeline(args.channel, steps)
        else:
            # Use traditional processing
            process_channel(args.channel, steps)
    else:
        start_msg = f"Starting the full automated process for all channels..."
        if use_timeline:
            start_msg += " (with timeline integration)"
        
        if has_logging:
            logger.info(start_msg)
        else:
            print(f"\n{start_msg}\n")
        
        # Process each channel sequentially
        for channel_number in range(1, 4):
            if use_timeline:
                process_channel_with_timeline(channel_number, steps)
            else:
                process_channel(channel_number, steps)
            
            # Add a delay between channels to avoid rate limits
            if channel_number < 3:
                delay_msg = f"Waiting {args.delay} seconds before processing next channel..."
                if has_logging:
                    logger.info(delay_msg)
                else:
                    print(f"\n{delay_msg}")
                time.sleep(args.delay)
        
        complete_msg = "All channels processed successfully!"
        if has_logging:
            logger.info(complete_msg)
        else:
            print(f"\n{complete_msg}")

if __name__ == "__main__":
    main()
