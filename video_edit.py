import json
import subprocess
import random
import os  # Keep for os.listdir for now
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import shutil
from openai import OpenAI
import traceback
import shlex
from config import config, get_channel_config
from file_manager import FileManager

# Initialize file manager
file_mgr = FileManager()

def create_placeholder_clip(output_path: Union[str, Path], duration: int = 60) -> None:
    """
    Create a simple placeholder video clip using ffmpeg.
    
    Args:
        output_path: Path to save the output file
        duration: Duration of the clip in seconds
    """
    # Convert string path to Path if needed
    if isinstance(output_path, str):
        output_path = Path(output_path)
        
    # Ensure the directory exists
    file_mgr.ensure_dir_exists(output_path.parent)
    
    # Get the voice file if it exists to determine proper duration
    voice_file = file_mgr.get_abs_path(config.file_paths.voice_file)
    if file_mgr.file_exists(voice_file):
        voice_duration = get_voice_duration(str(voice_file))
        if voice_duration and voice_duration > 10:
            # Add buffer to voice duration
            duration = int(voice_duration) + 5
            print(f"Setting placeholder duration to match voice: {duration}s")
    
    # Set the proper resolution for the placeholder
    width = 1080
    height = 1920
    
    # Create a command to generate a simple test pattern
    try:
        print(f"Creating placeholder video at {output_path}...")
        # Simple command to create a color test pattern
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=blue:s={width}x{height}:d={duration}",
            "-vf", f"drawtext=text='Placeholder Video':fontcolor=white:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-t", str(duration),
            str(output_path)
        ], check=True)
        print("Placeholder video created successfully")
    except Exception as e:
        print(f"Error creating placeholder video: {e}")
        # Try a simpler approach
        try:
            print("Trying alternative method...")
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=blue:s={width}x{height}:d={duration}",
                "-c:v", "libx264",
                "-preset", "fast",
                str(output_path)
            ], check=True)
            print("Basic placeholder video created successfully")
        except Exception as e:
            print(f"Error creating basic placeholder video: {e}")
            raise

def load_clips_metadata() -> List[Dict]:
    """
    Load and parse the clips metadata from the configured CSV metadata file.
    If CSV fails, scan directory for actual clips.
    
    Returns:
        List[Dict]: A list of clip metadata dictionaries
    """
    clips_file = file_mgr.get_abs_path(config.file_paths.clips_metadata_file)
    clips_dir = file_mgr.get_abs_path(config.file_paths.clips_directory)
    
    # Create sample_clips directory if it doesn't exist
    sample_clips_dir = clips_dir / "sample_clips"
    file_mgr.ensure_dir_exists(sample_clips_dir)
    
    # Create a placeholder clip
    placeholder_path = sample_clips_dir / "placeholder.mp4"
    if not file_mgr.file_exists(placeholder_path):
        try:
            create_placeholder_clip(placeholder_path)
        except Exception as e:
            print(f"Warning: Failed to create placeholder clip: {e}")
    
    # First try loading from CSV
    clips = []
    csv_clips = []
    
    try:
        # Use FileManager to read the file
        content_str = file_mgr.read_text(clips_file)
        if content_str is not None:
            # Parse CSV format
            lines = content_str.strip().split('\n')
            if len(lines) >= 2:  # At least header + one data row
                # Get header for column indexing
                header = lines[0].split(',')
                
                # Find indices of required columns
                try:
                    path_idx = header.index('path')
                    # prompt_idx = header.index('prompt')
                    # aspect_ratio_idx = header.index('aspect_ratio')
                    duration_idx = header.index('duration')
                    # labels_idx = header.index('labels')
                    image_caption_idx = header.index('image_1_caption')
                    
                    
                    
                    print(f"Found {len(lines) - 1} clip entries in the CSV file")
                    
                    # Process each data row
                    for i in range(1, len(lines)):
                        line = lines[i]
                        if not line.strip():
                            continue
                            
                        # Handle CSV commas within quoted fields
                        parts = []
                        in_quotes = False
                        current_part = ''
                        
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                parts.append(current_part)
                                current_part = ''
                            else:
                                current_part += char
                        
                        # Add the last part
                        parts.append(current_part)
                        
                        # Skip if we don't have enough columns
                        if len(parts) <= max(path_idx, image_caption_idx, duration_idx):
                            print(f"Warning: Skipping line {i+1}, insufficient columns: {line}")
                            continue
                        
                        # Extract values from CSV
                        path_value = parts[path_idx].strip()
                        # prompt_value = parts[prompt_idx].strip().strip('"')
                        # short_prompt_value = parts[short_prompt_idx].strip().strip('"')
                        # aspect_ratio_value = parts[aspect_ratio_idx].strip()
                        duration_value = parts[duration_idx].strip()
                        # labels_value = parts[labels_idx].strip().strip('"')
                        image_caption = parts[image_caption_idx].strip().strip('"')
                        
                        # Extract filename from path and handle full path correctly
                        path_obj = Path(path_value)
                        
                        # Store both the name and the full path in the metadata
                        file_name = path_obj.name
                        file_path = str(path_obj)  # Keep the full path including directory
                        
                        # Process duration value
                        try:
                            # Handle duration in format like "10s"
                            if duration_value.lower().endswith('s'):
                                duration_seconds = int(duration_value.lower().rstrip('s'))
                            else:
                                duration_seconds = config.video_edit.default_clip_duration
                        except ValueError:
                            duration_seconds = config.video_edit.default_clip_duration
                            print(f"Warning: Could not parse duration '{duration_value}' for clip '{file_name}'")
                        
                        # Create clip metadata dictionary
                        clip = {
                            'name': file_name,
                            'path': file_path,
                            # 'description': prompt_value,
                            'duration': duration_seconds,
                            # 'notes': labels_value,
                            # 'aspect_ratio': aspect_ratio_value,
                            'image_caption': image_caption
                        }
                        
                        # Add the clip to the list
                        csv_clips.append(clip)
                except ValueError as e:
                    print(f"Error parsing CSV columns: {e}")
    except Exception as e:
        print(f"Error loading CSV metadata: {e}")
        traceback.print_exc()
    
    print(f"Loaded {len(csv_clips)} clips from CSV metadata")
    
    # Now scan the actual clips directory for all available video files
    print(f"Scanning clips directory at {clips_dir} for video files...")

    # Merge CSV and scanned clips, with CSV taking precedence for metadata
    # but ensuring all actual files are included
    
    # First add all CSV clips that have validated actual files
    for clip in csv_clips:
        if 'path' in clip:
            full_path = clip['path']
            clip_path = file_mgr.get_abs_path(full_path)
        else:
            clip_path = clips_dir / clip['name']
        
        # Try to find the video file
        video_file = file_mgr.find_video_file(clip_path)
        if video_file:
            clip['full_path'] = str(video_file)
            clips.append(clip)
        else:
            # Try the clips directory directly
            alt_path = clips_dir / clip['name'] 
            video_file = file_mgr.find_video_file(alt_path)
            if video_file:
                clip['full_path'] = str(video_file)
                clips.append(clip)
    
    # Keep track of which files we've already added
    added_paths = set(clip.get('full_path', '') for clip in clips)
    
    # If still no valid clips, add placeholder
    if not clips:
        print("No valid clips found in CSV or directory scan. Using placeholder.")
        clips = [{
            'name': "sample_clips/placeholder.mp4",
            'description': "Placeholder video for testing",
            'duration': 10,
            'notes': "Auto-generated placeholder"
        }]
    
    # Show summary of available clips
    print(f"Final clip count: {len(clips)} valid clips")
    for i, clip in enumerate(clips[:5]):  # Show only first 5 for brevity
        print(f"Clip {i+1}: {clip['name']}, Duration: {clip['duration']}s")
    
    if len(clips) > 5:
        print(f"... and {len(clips) - 5} more clips")
        
    return clips

def get_script_segments(channel_number: Optional[int] = None) -> str:
    """
    Load the script content from the configured file path
    
    Args:
        channel_number (Optional[int]): Channel number to use for configuration
        
    Returns:
        str: The content of the script file
    """
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Try multiple script file paths with priority on channel-specific paths
    script_file_paths = [
        file_mgr.get_script_path(channel_number, config.file_paths.script_file),
        # Last resort: global script path
        file_mgr.get_abs_path(config.file_paths.script_file),
    ]
    
    print(f"Looking for script in these locations:")
    for path in script_file_paths:
        print(f"  - {path}")
    
    script_content = None
    used_path = None
    
    # Try each path until we find one that works
    for path in script_file_paths:
        content = file_mgr.read_text(path)
        if content is not None:
            script_content = content
            used_path = path
            print(f"Successfully read script from {used_path}")
            break
    
    if script_content is None:
        print(f"Error: Could not read script file from any of these paths: {script_file_paths}")
        # Provide fallback content
        script_content = """The Rise of AI Agents: 
        
        Artificial Intelligence agents are rapidly transforming how we work. These software entities can autonomously complete complex tasks with minimal human supervision. Unlike traditional AI systems, AI agents can make decisions, use tools, and execute action sequences to accomplish goals.
        
        Major tech companies are advancing this technology quickly. These capabilities will change knowledge work forever."""
        print("Using fallback script content")
        
    return script_content

def get_voice_duration(voice_file: str) -> float:
    """Uses ffprobe to get the duration (in seconds) of the generated voice audio."""
    try:
        result = subprocess.run(
            [
              "ffprobe", "-v", "error",
              "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1",
              voice_file
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        duration_str = result.stdout.strip()
        return float(duration_str)
    except Exception as e:
        print("Error obtaining voice duration:", e)
        return None

def get_num_segments(srt_file: Optional[str] = None, channel_number: Optional[int] = None) -> int:
    """
    Determine the number of subtitle segments in the SRT file
    
    Args:
        srt_file (Optional[str]): Path to the SRT file, or None to use the configured default
        channel_number (Optional[int]): Channel number to use for configuration
        
    Returns:
        int: Number of subtitle segments
    """
    if srt_file is None:
        if channel_number is not None:
            srt_file = str(file_mgr.get_caption_path(channel_number, config.file_paths.captions_file))
        else:
            srt_file = file_mgr.get_abs_path(config.file_paths.captions_file)
    
    print(f"Reading subtitles from: {srt_file}")
    srt_content = file_mgr.read_text(srt_file)
    if srt_content is None:
        print(f"Error: Could not read SRT file: {srt_file}")
        return 0
    
    subtitles = [seg for seg in srt_content.strip().split('\n\n') if seg.strip() != '']
    print(f"Found {len(subtitles)} subtitle segments")
    return len(subtitles)

def match_clips_to_script(script: str, clips: List[Dict], target_duration: float = None, channel_number: Optional[int] = None) -> List[Dict]:
    """
    Use OpenAI to match clips to script segments based on SRT timestamps and content.
    With fallback to simple matching if AI fails.
    
    Args:
        script (str): The script content
        clips (List[Dict]): List of clip metadata
        target_duration (float, optional): Target duration of the video
        
    Returns:
        List[Dict]: A list of clip segments matched to script segments
    """
    # Initialize OpenAI client with API key from config
    client = OpenAI(api_key=config.openai.api_key)
    
    # Create a set of available clip names for validation
    available_clips = {clip['name'] for clip in clips}
    
    # Read SRT file for timing information
    srt_file = file_mgr.get_abs_path(f"outputs/channel_{channel_number}/{config.file_paths.captions_file}")
    srt_content = file_mgr.read_text(srt_file)
    
    if srt_content is None:
        print(f"Error: Could not read SRT file: {srt_file}")
        return []
    
    # Create a placeholder in case the SRT has issues
    placeholder_clip_name = "sample_clips/placeholder.mp4"
    
    # If there's no SRT but there's a voice file, use that for timing
    if not srt_content and target_duration:
        print("SRT file missing or empty but target duration available. Creating simple sequence.")
        return [{
            'clip_name': placeholder_clip_name,
            'start_time': 0,
            'duration': target_duration,
            'script_segment': script[:100] + "..."  # First part of script
        }]
    
    # Parse SRT to get actual segment durations and scripts
    srt_segments = [seg.strip() for seg in srt_content.split('\n\n') if seg.strip()]
    segment_timings = []
    segment_texts = []
    
    for segment in srt_segments:
        lines = segment.split('\n')
        if len(lines) >= 3:  # Num, timing, text
            times = lines[1].split(' --> ')
            if len(times) == 2:
                start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(times[0].replace(',', '.').split(':'))))
                end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(times[1].replace(',', '.').split(':'))))
                duration = end - start
                segment_timings.append(duration)
                
                # Get the text part (could be multiple lines)
                text = ' '.join(lines[2:])
                segment_texts.append(text)
                
    # If no valid segments found, create at least one using total duration
    if not segment_timings and target_duration:
        print("No valid segment timings found. Using total duration.")
        segment_timings = [target_duration]
        segment_texts = [script[:100] + "..."]
    
    # Modify the clips info to include ONLY available clips
    max_clip_duration = config.video_edit.max_clip_duration
    clips_info = "\n".join([
        f"Clip: {c['name']}\nImage Caption: {c['image_caption']}\n"
        f"Duration: {c['duration']}s\n"
        f"Possible start times: 0 to {max(0, c['duration'] - max_clip_duration)} seconds\n"
        for c in clips
    ])
    
    target_duration_text = ""
    if target_duration is not None:
        target_duration_text = f"\nTotal Generated Voice Duration: {int(target_duration)} seconds."
    
    # Extract the main script segments for better analysis
    script_excerpt = script[:500] if len(script) > 500 else script
    srt_excerpt = '\n'.join(segment_texts[:10]) if len(segment_texts) > 10 else '\n'.join(segment_texts)
    
    prompt = f"""Given these available video clips along with their metadata:
    
{clips_info}
{target_duration_text}

And this SRT file with timestamps and script segments:

{srt_content}

And this script excerpt to understand the theme:
{script_excerpt}

Your task is to create a sequence of clips that best matches the voiceover content and timing. When selecting clip segments:
- First, analyze the script to understand the main theme and topic
- Choose clips whose descriptions or notes MATCH the content of each script segment
- Use the EXACT SRT timestamps to ensure clips align with the voiceover timing
- Each clip segment MUST match the exact duration of its corresponding SRT segment
- If a segment needs multiple clips, divide the time equally between them
- For each clip selection, start time must be within the possible start times range
- Ensure the first 15 seconds use at least 3 different clips for visual variety
- Prioritize high-quality clips that match the topic and mood of the script segment
- IMPORTANT: ONLY use clip names from the available clips provided above
- CRITICAL: The clip_name field MUST exactly match one of the clips listed above
- **Never use the same clip more than once**
- **Never use a clip less than 2 seconds**

Return a JSON array where each object has:
- clip_name: the EXACT filename of one of the clips listed above (*.mp4)
- start_time: when to start using the clip (in seconds from the clip's beginning)
- duration: length of the clip segment (MUST match the SRT segment duration)
- script_segment: the part of the script that this clip should align with
- explanation: the reasoning behind the logic of choosing that clip for that particular segment among other options
- suggestion: more direct alternative to be shown to the audience that would match the segment and the general theme of the video better in a descriptive way that could be used as a prompt for generating images

Format the response as valid JSON only, no additional text.

**It's CRUCIAL to keep JSON as expected otherwise it would cause an error**"""
    
    try:
        response = client.chat.completions.create(
            model=config.openai.video_edit_model,
            messages=[
                {
                    "role": "developer", 
                    "content": "You are a video editing assistant with expertise in aligning clip metadata with script content and timing for fast-paced, dynamic edits."
                },
                {"role": "user", "content": prompt}
            ]
        )
        
        clip_sequence = json.loads(response.choices[0].message.content)
        with open("json_response.json", "w") as output_json:
            output_json.write(response.model_dump_json())

        # Validate clips and filter out any that don't exist
        validated_sequence = []
        
        # Create placeholder clip info
        placeholder_clip_name = "sample_clips/placeholder.mp4"
        
        # Print what clip names are available for debugging
        print(f"Available clips: {', '.join(available_clips)[:]}...")
        invalid_clips = []
        
        for segment in clip_sequence:
            if segment['clip_name'] in available_clips:
                validated_sequence.append(segment)
            else:
                invalid_clips.append(segment['clip_name'])
                print(f"Warning: Clip '{segment['clip_name']}' not found. Using placeholder.")
                # Create a new segment with the placeholder clip
                placeholder_segment = segment.copy()
                placeholder_segment['clip_name'] = placeholder_clip_name
                placeholder_segment['start_time'] = 0  # Use beginning of placeholder
                validated_sequence.append(placeholder_segment)
        
        if invalid_clips:
            print(f"WARNING: {len(invalid_clips)} invalid clip names were provided by the AI: {', '.join(invalid_clips)}")
            
            # If too many invalid clips, use the manual matching approach
            if len(invalid_clips) > len(clip_sequence) / 2:
                print("Too many invalid clips. Using manual matching instead.")
        
        # If no valid segments found at all, create basic segments with placeholder
        if not validated_sequence and segment_timings:
            print("No valid clips. Using manual matching.")
        
        # Ensure the durations match the SRT segments
        for i, segment in enumerate(validated_sequence):
            if i < len(segment_timings):
                segment['duration'] = segment_timings[i]
        
        return validated_sequence
    
    except (json.JSONDecodeError, Exception) as e:
        raise Exception(f"Error in AI clip matching: {e}")
        


def enforce_clip_duration(clip_sequence: List[Dict]) -> List[Dict]:
    """
    Ensure that every clip segment's duration is within the configured minimum and maximum.
    
    Args:
        clip_sequence (List[Dict]): The sequence of clips
        
    Returns:
        List[Dict]: The adjusted clip sequence
    """
    max_duration = config.video_edit.max_clip_duration
    min_duration = 2.0  # Minimum 2 seconds for any clip
    
    for clip in clip_sequence:
        if clip['duration'] > max_duration:
            print(f"Adjusting clip '{clip['clip_name']}' duration from {clip['duration']}s to {max_duration}s for faster-paced edits.")
            clip['duration'] = max_duration
        elif clip['duration'] < min_duration:
            print(f"Adjusting clip '{clip['clip_name']}' duration from {clip['duration']}s to {min_duration}s for better viewing experience.")
            clip['duration'] = min_duration
    return clip_sequence

def validate_clip_sequence(clip_sequence: List[Dict], clips_metadata: List[Dict]) -> List[Dict]:
    """Validate and adjust clip start times and durations to ensure they're within valid ranges,
    avoiding reusing the same clips and ensuring proper transitions."""
    clips_dict = {clip['name']: clip['duration'] for clip in clips_metadata}
    used_segments = {}  # clip_name -> list of (start_time, end_time) tuples
    used_clips = set()  # Track which clips have been used
    
    print(f"Total segments to process: {len(clip_sequence)}")
    total_duration_before = sum(segment['duration'] for segment in clip_sequence)
    print(f"Total duration before validation: {total_duration_before:.2f} seconds")
    
    # First pass to identify all clips
    available_clips = set([clip['name'] for clip in clips_metadata])
    
    # Process each segment
    for i, segment in enumerate(clip_sequence):
        clip_name = segment['clip_name']
        total_duration = clips_dict.get(clip_name, 0)
        original_duration = segment['duration']  # Store the original duration
        
        print(f"\nProcessing segment {i+1}/{len(clip_sequence)}")
        print(f"Clip: {clip_name}, Original Duration: {original_duration:.2f}s")
        
        # If we've already used this clip and there are alternatives, try to find a different one
        if clip_name in used_clips and len(available_clips - used_clips) > 0:
            print(f"Warning: Clip '{clip_name}' has already been used. Trying to find an alternative.")
            
            # Find available alternatives
            alternatives = list(available_clips - used_clips)
            if alternatives:
                new_clip = random.choice(alternatives)
                clip_name = new_clip
                segment['clip_name'] = new_clip
                total_duration = clips_dict.get(clip_name, 0)
                print(f"Replaced with alternative clip: {clip_name}")
        
        # Track used clips
        used_clips.add(clip_name)
        
        # Initialize used segments tracking for this clip if not already exists
        if clip_name not in used_segments:
            used_segments[clip_name] = []
        
        # For clips longer than 11 seconds, try to find an unused segment
        if total_duration > 11:
            max_attempts = 10
            attempt = 0
            found_valid_segment = False
            
            while attempt < max_attempts:
                # Generate a random start time
                max_start = max(0, total_duration - original_duration)
                proposed_start = random.uniform(0, max_start)
                proposed_end = proposed_start + original_duration
                
                # Check if this segment overlaps with any used segments
                overlap = False
                for used_start, used_end in used_segments[clip_name]:
                    if not (proposed_end < used_start or proposed_start > used_end):
                        overlap = True
                        break
                
                if not overlap and proposed_end <= total_duration:
                    segment['start_time'] = proposed_start
                    segment['duration'] = original_duration  # Maintain original duration
                    used_segments[clip_name].append((proposed_start, proposed_end))
                    found_valid_segment = True
                    print(f"Found valid segment: Start={proposed_start:.2f}s, Duration={original_duration:.2f}s")
                    break
                
                attempt += 1
            
            # If we couldn't find an unused segment, try to find any valid segment
            if not found_valid_segment:
                max_start = max(0, total_duration - original_duration)
                segment['start_time'] = random.uniform(0, max_start)
                segment['duration'] = original_duration  # Maintain original duration
                print(f"Using fallback segment: Start={segment['start_time']:.2f}s, Duration={original_duration:.2f}s")
        else:
            # For shorter clips, just use random start time
            max_start = max(0, total_duration - original_duration)
            segment['start_time'] = random.uniform(0, max_start)
            segment['duration'] = original_duration  # Maintain original duration
            print(f"Short clip segment: Start={segment['start_time']:.2f}s, Duration={original_duration:.2f}s")
        
        # Final safety check to ensure we don't exceed clip boundaries
        if segment['start_time'] + segment['duration'] > total_duration:
            segment['start_time'] = max(0, total_duration - original_duration)
            print(f"Applied safety adjustment: Start={segment['start_time']:.2f}s, Duration={original_duration:.2f}s")
            
    total_duration_after = sum(segment['duration'] for segment in clip_sequence)
    print(f"\nTotal duration after validation: {total_duration_after:.2f} seconds")
    
    return clip_sequence

def create_video_sequence(clip_sequence: List[Dict], clips_metadata: List[Dict] = None, channel_number: Optional[int] = None) -> bool:
    """
    Use ffmpeg to concatenate the selected clip segments into a video with proper transitions.
    
    Args:
        clip_sequence (List[Dict]): The sequence of clips to concatenate
        clips_metadata (List[Dict], optional): The full metadata for all available clips
        channel_number (int, optional): Channel number to use, or None to use default
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Get file paths from configuration
    clips_dir = file_mgr.get_abs_path(config.file_paths.clips_directory)
    
    # Use channel-specific output path
    output_dir = file_mgr.get_channel_output_path(channel_number)
    output_video = output_dir / config.file_paths.output_video_file
    
    # If clips_metadata not provided, use load_clips_metadata() to get it
    clips = clips_metadata if clips_metadata is not None else load_clips_metadata()
    
    # Ensure the output directory exists
    file_mgr.ensure_dir_exists(output_video.parent)
    
    # Check if we have any valid clips
    if not clips or not clip_sequence:
        print("Error: No valid clips or clip sequence available")
        # Create a placeholder video for the entire sequence
        create_placeholder_clip(output_video, 60)
        return True
    
    # Ensure temp directory exists
    temp_dir = file_mgr.get_abs_path("temp_files")
    file_mgr.ensure_dir_exists(temp_dir)
    
    try:
        # Process each clip segment individually first
        segments_list = []
        
        for i, clip in enumerate(clip_sequence):
            # Use the full validated path with extension if available
            clip_name = clip['clip_name']
            
            # First check if there's a stored full path from earlier validation
            # Store the matching clip object so we can access all its metadata
            matching_clip = None
            for c in clips:
                if c.get('name') == clip_name and 'full_path' in c:
                    clip_path = Path(c['full_path'])
                    matching_clip = c
                    print(f"Found matching clip with full_path: {clip_path}")
                    break
            else:
                # Otherwise look for the file with potential extensions
                # Try three different approaches to find the video:
                # 1. Look in the clips directory for the name as-is
                clip_path = clips_dir / clip_name
                video_file = file_mgr.find_video_file(clip_path)
                
                if video_file:
                    clip_path = video_file
                    print(f"Found clip using clips_dir path: {clip_path}")
                else:
                    # 2. Look directly in video directory (handling potential path prefixes in CSV)
                    video_file = file_mgr.find_video_file(Path("video") / clip_name)
                    if video_file:
                        clip_path = video_file
                        print(f"Found clip using video/ prefix: {clip_path}")
                    else:
                        # 3. Try with explicit .mp4 extension
                        mp4_path = clips_dir / f"{clip_name}"
                        if not mp4_path.suffix:  # If no extension, add .mp4
                            mp4_path = clips_dir / f"{clip_name}.mp4"
                        video_file = file_mgr.find_video_file(mp4_path)
                        if video_file:
                            clip_path = video_file
                            print(f"Found clip using .mp4 extension: {clip_path}")
                        else:
                            # Last resort: use the path as given
                            clip_path = clips_dir / clip_name
                            print(f"Using last resort path: {clip_path}")
            
            start_time = clip.get('start_time', 0)
            duration = clip['duration']
            
            # Output path for this segment
            segment_output = temp_dir / f"segment_{i:03d}.mp4"
            segments_list.append(str(segment_output))
            
            # Check if clip exists
            if not file_mgr.file_exists(clip_path):
                print(f"Error: Clip file doesn't exist: {clip_path}")
                # Create a placeholder for this segment
                create_placeholder_clip(segment_output, duration)
                continue
                
            try:
                # Scaling parameters for consistent output
                output_width = 1080  
                output_height = 1920
                
                # Simple scaling without padding to avoid errors
                scale_filter = f'scale={output_width}:{output_height},setsar=1:1'
                
                # Keep track of the original aspect ratio for reference
                aspect_ratio = clip.get('aspect_ratio', '9:16')
                print(f"Processing clip with aspect ratio: {aspect_ratio} using simple scaling")
                
                # Extract segment with exact duration and proper scaling
                subprocess.run([
                    'ffmpeg',
                    '-y',
                    '-ss', str(start_time),
                    '-i', str(clip_path),
                    '-t', str(duration),
                    '-vf', scale_filter,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '22',
                    '-r', '30',
                    '-pix_fmt', 'yuv420p',
                    str(segment_output)
                ], check=True)
                print(f"Created segment {i}: {segment_output} (duration: {duration:.2f}s)")
                
                # Verify the segment was created
                if not segment_output.exists():
                    print(f"Warning: Segment {i} was not created at {segment_output}")
                    create_placeholder_clip(segment_output, duration)
                
            except subprocess.CalledProcessError as e:
                print(f"Error creating segment {i}: {e}")
                # Create a placeholder for this segment as fallback
                create_placeholder_clip(segment_output, duration)
        
        # Create a concat file for the processed segments
        concat_file = temp_dir / "concat_list.txt"
        
        # Verify which segments actually exist before adding to concat list
        valid_segments = []
        for segment_path in segments_list:
            if Path(segment_path).exists():
                valid_segments.append(segment_path)
            else:
                print(f"Warning: Segment {segment_path} does not exist and won't be included")
                
        if not valid_segments:
            print("No valid segments found. Creating a placeholder video instead.")
            create_placeholder_clip(output_video, 60)
            return True
            
        # Create concat file with absolute paths
        concat_content = "\n".join([f"file '{seg}'" for seg in valid_segments])
        file_mgr.write_text(concat_file, concat_content)
        
        print(f"Concat file contents: {concat_content}")
        
        # Concatenate all the standardized segments
        try:
            print(f"Concatenating segments into final video: {output_video}")
            
            # Use simpler approach that's more reliable
            if len(valid_segments) == 1:
                # If only one segment, just copy it
                print("Only one segment. Copying directly to output.")
                shutil.copy2(valid_segments[0], output_video)
            else:
                # Use concat demuxer for multiple segments
                subprocess.run([
                    'ffmpeg',
                    '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    '-c', 'copy',  # We can use copy here since all segments are already properly encoded
                    str(output_video)
                ], check=True)
                
            print(f"Video sequence created successfully")
            
            # Verify the file was actually created
            if not file_mgr.file_exists(output_video):
                print(f"Warning: Output file not found at {output_video} despite successful command")
                # Create placeholder as fallback
                print("Creating placeholder video as fallback...")
                create_placeholder_clip(output_video, 60)
                print(f"Created placeholder video at {output_video}")
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"Error creating final video sequence: {e}")
            print(f"Error details: {str(e)}")
            
            # Create placeholder as fallback
            try:
                print("Creating placeholder video as fallback...")
                placeholder_duration = 60  # seconds
                create_placeholder_clip(output_video, placeholder_duration)
                print(f"Created placeholder video at {output_video}")
                return True
            except Exception as e2:
                print(f"Error creating placeholder video: {e2}")
                return False
    
    except Exception as e:
        print(f"Unexpected error in create_video_sequence: {e}")
        traceback.print_exc()
        # Create placeholder as ultimate fallback
        create_placeholder_clip(output_video, 60)
        return False

def merge_voice_with_video(video_path: str = None, voice_path: str = None, output_path: str = None, channel_number: Optional[int] = None) -> bool:
    """
    Merge the generated voice audio with the video and add background music as a second audio layer.
    Ensure the final video duration matches the voice track length.
    
    Args:
        video_path (str, optional): Path to the video file, or None to use config default
        voice_path (str, optional): Path to the voice file, or None to use config default
        output_path (str, optional): Path for the output file, or None to use config default
        channel_number (int, optional): Channel number to use, or None to use default
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Store the original video path to handle temporary extended videos
    original_video_path = video_path
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Use channel-specific paths if parameters are not provided
    if video_path is None:
        output_dir = file_mgr.get_channel_output_path(channel_number)
        video_path = str(output_dir / config.file_paths.output_video_file)
    if voice_path is None:
        voice_path = str(file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/","")))
    if output_path is None:
        output_dir = file_mgr.get_channel_output_path(channel_number)
        output_path = str(output_dir / config.file_paths.final_video_file)
    
    # Ensure the output directory exists
    file_mgr.ensure_dir_exists(Path(output_path).parent)
    
    # Check if input files exist
    if not file_mgr.file_exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Creating placeholder video since input video is missing")
        try:
            create_placeholder_clip(output_path, 60)
            print(f"Created placeholder video at {output_path}")
            return True
        except Exception as e:
            print(f"Error creating placeholder: {e}")
            return False
    
    # Check if voice file exists
    if not file_mgr.file_exists(voice_path):
        print(f"Voice file not found: {voice_path}")
        print("Copying video file to output without voice")
        try:
            file_mgr.copy_file(video_path, output_path)
            print(f"Video copied to {output_path}")
            return True
        except Exception as e:
            print(f"Error copying video: {e}")
            # Create placeholder as last resort
            try:
                create_placeholder_clip(output_path, 60)
                return True
            except:
                return False
    
    # Get voice duration to ensure video covers the entire voice track
    voice_duration = get_voice_duration(voice_path)
    if voice_duration is None:
        print("Warning: Could not determine voice duration")
        voice_duration = 60  # Default fallback
    else:
        print(f"Voice duration: {voice_duration:.2f} seconds")
    
    # Get video duration
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        video_duration = float(result.stdout.strip())
        print(f"Video duration: {video_duration:.2f} seconds")
    except Exception as e:
        print(f"Error obtaining video duration: {e}")
        video_duration = None
    
    # Get audio volume settings from config
    voice_volume = config.video_edit.voice_volume
    bgm_volume = config.video_edit.background_music_volume
    
    # Get background music folder path from config
    bgm_folder = file_mgr.get_abs_path(config.file_paths.background_music_directory)
    
    # Create sample music folder if it doesn't exist
    file_mgr.ensure_dir_exists(bgm_folder)
    
    # Create a sample music file if none exists
    if not file_mgr.list_files(bgm_folder, "*.mp3"):
        print("No background music files found. Creating a sample tone...")
        sample_tone_path = bgm_folder / "background_music.mp3"
        try:
            # Create a 30-second sine wave tone (actually audible, not silent)
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi", 
                "-i", "sine=frequency=440:sample_rate=44100:duration=30",
                "-filter_complex", "afade=t=in:st=0:d=2,afade=t=out:st=28:d=2",
                "-ar", "44100",
                "-ac", "2",
                "-b:a", "192k",
                str(sample_tone_path)
            ], check=True)
            print(f"Created audible background tone at {sample_tone_path}")
        except Exception as e:
            print(f"Could not create sample tone: {e}")
    
    # Use file_mgr to list files in the background music folder
    bgm_files = [str(path) for path in file_mgr.list_files(bgm_folder, "*.mp3")]
    
    # Handle case where video is shorter than voice
    if video_duration and voice_duration and video_duration < voice_duration:
        print(f"Warning: Video ({video_duration:.2f}s) is shorter than voice ({voice_duration:.2f}s). Extending video...")
        
        # Create a persistent temporary file for extended video
        temp_dir = file_mgr.get_abs_path("temp_files")
        file_mgr.ensure_dir_exists(temp_dir)
        import uuid  # Import here so we don't redefine it later
        ext_temp_name = f"temp_ext_{uuid.uuid4().hex}.mp4"
        extended_video_path = temp_dir / ext_temp_name
        
        try:
            # Instead of using tpad which creates a still frame, let's use loop
            # This will repeat the video from the beginning rather than freezing on last frame
            # Calculate number of loops needed to cover the voice duration
            loops_needed = int(voice_duration / video_duration) + 1
            
            subprocess.run([
                "ffmpeg", "-y",
                "-stream_loop", str(loops_needed), 
                "-i", video_path,
                "-t", str(voice_duration + 1),  # Add a safety margin
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                str(extended_video_path)
            ], check=True)
            
            print(f"Extended video created successfully by looping (total duration: {voice_duration + 1:.2f}s)")
            # Use the extended video for merging
            video_path = str(extended_video_path)
            # We'll let the file stay around until the entire function completes
            # since we're using video_path for the next step
        except subprocess.CalledProcessError as e:
            print(f"Error extending video by looping: {e}")
            # Fall back to the old approach if looping fails
            try:
                # Extend the video by looping the last frame to match voice duration
                # Add a 1-second safety margin
                extend_duration = voice_duration - video_duration + 1
                
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-filter_complex", 
                    f"[0:v]tpad=stop_mode=clone:stop_duration={extend_duration}[extended]",
                    "-map", "[extended]",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "22",
                    str(extended_video_path)
                ], check=True)
                
                print(f"Extended video created successfully using freeze frame (total duration: {voice_duration + 1:.2f}s)")
                # Use the extended video for merging
                video_path = str(extended_video_path)
            except subprocess.CalledProcessError as e:
                print(f"Error extending video using freeze frame: {e}")
                # Continue with original video if extension fails
            
            # Clean up the file if it exists but is invalid
            if extended_video_path.exists() and not os.path.getsize(str(extended_video_path)) > 0:
                try:
                    extended_video_path.unlink()
                    print(f"Cleaned up invalid extended video file: {extended_video_path}")
                except Exception as e2:
                    print(f"Warning: Could not remove temporary file {extended_video_path}: {e2}")
    
    # Simplify by first merging video with voice, then add background music if available
    try:
        print(f"Merging voice with video: {output_path}")
        
        # Create a persistent temporary file (not using context manager to avoid early deletion)
        temp_dir = file_mgr.get_abs_path("temp_files")
        file_mgr.ensure_dir_exists(temp_dir)
        import uuid
        temp_name = f"temp_voice_merge_{uuid.uuid4().hex}.mp4"
        temp_output = temp_dir / temp_name
        
        try:
            # Add voice to video
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", voice_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(temp_output)
            ], check=True)
            
            print("Successfully merged voice with video")
            
            # Now add background music (always add it if music exists)
            if bgm_files:
                bgm_file = random.choice(bgm_files)
                print(f"Selected background music: {bgm_file}")
                
                # Mix the background music at a lower volume
                try:
                    # Print detailed info about what we're doing
                    print(f"Adding background music '{bgm_file}' at volume {bgm_volume}")
                    
                    # First verify the background music file exists
                    if not os.path.exists(bgm_file):
                        print(f"Warning: Background music file '{bgm_file}' not found")
                        # Look in the correct directory if needed
                        alt_bgm_path = file_mgr.get_abs_path("background_music/background_music.mp3")
                        if os.path.exists(str(alt_bgm_path)):
                            bgm_file = str(alt_bgm_path)
                            print(f"Using alternative background music path: {bgm_file}")
                    
                    # More robust command with better error handling and detailed debugging
                    process = subprocess.run([
                        "ffmpeg",
                        "-y",
                        "-i", str(temp_output),
                        "-i", bgm_file,
                        "-filter_complex", 
                        f"[0:a]volume={voice_volume}[main];[1:a]volume={bgm_volume},aloop=loop=-1:size=2s[bgm];[main][bgm]amix=inputs=2:duration=first[aout]",
                        "-map", "0:v",
                        "-map", "[aout]",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",  # Higher audio bitrate for better quality
                        "-shortest",
                        output_path
                    ], check=True)
                    
                    print("Successfully added background music")
                    
                    # Verify that the output file was created with the expected duration
                    if not file_mgr.file_exists(output_path):
                        raise Exception(f"Output file was not created: {output_path}")
                    
                    # Clean up temp file
                    if temp_output.exists():
                        temp_output.unlink()
                        print(f"Cleaned up temporary file: {temp_output}")
                    
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Error adding background music: {e}")
                    print(f"Error output: {e.stderr}")
                    # If background music fails, use the voice-only version
                    file_mgr.copy_file(temp_output, output_path)
                    print("Using voice-only version as fallback")
                    
                    # Clean up temp file
                    if temp_output.exists():
                        temp_output.unlink()
                        print(f"Cleaned up temporary file: {temp_output}")
                    
                    return True
            else:
                # Just use the voice version
                file_mgr.copy_file(temp_output, output_path)
                print("No background music files found. Using voice-only version.")
                
                # Clean up temp file
                if temp_output.exists():
                    temp_output.unlink()
                    print(f"Cleaned up temporary file: {temp_output}")
                
                return True
        finally:
            # Make sure we always clean up the temp file
            if temp_output.exists():
                try:
                    temp_output.unlink()
                    print(f"Cleaned up temporary file: {temp_output}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {temp_output}: {e}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error merging voice with video: {e}")
        # Try a simpler approach as fallback
        try:
            print("Trying simpler approach...")
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", voice_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path
            ], check=True)
            print("Voice and video merged successfully with simpler approach")
        except subprocess.CalledProcessError as e2:
            print(f"Error with simpler approach: {e2}")
            # Just copy the video as last resort
            try:
                file_mgr.copy_file(video_path, output_path)
                print(f"Copied video without audio as last resort")
                return True
            except Exception as e3:
                print(f"Failed to copy video: {e3}")
                # Try creating a placeholder as absolute last resort
                try:
                    create_placeholder_clip(output_path, 60)
                    print(f"Created placeholder as last resort")
                    
                    # Clean up extended video if it exists and is different from original
                    if video_path != original_video_path and Path(video_path).exists():
                        try:
                            Path(video_path).unlink()
                            print(f"Cleaned up temporary extended video: {video_path}")
                        except Exception as e:
                            print(f"Warning: Could not remove extended video: {e}")
                    
                    return True
                except:
                    return False
    except Exception as e:
        print(f"Unexpected error during audio processing: {e}")
        # Just copy the video as last resort
        try:
            file_mgr.copy_file(video_path, output_path)
            print(f"Copied video without audio due to unexpected error")
            return True
        except Exception as e2:
            print(f"Failed to copy video: {e2}")
            # Try creating a placeholder as absolute last resort
            try:
                create_placeholder_clip(output_path, 60)
                print(f"Created placeholder as last resort")
                
                # Clean up extended video if it exists and is different from original
                if video_path != original_video_path and Path(video_path).exists():
                    try:
                        Path(video_path).unlink()
                        print(f"Cleaned up temporary extended video: {video_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove extended video: {e}")
                
                return True
            except:
                return False

def burn_subtitles(video_path: str = None, srt_path: str = None, output_path: str = None, channel_number: Optional[int] = None) -> bool:
    """
    Burn subtitles from the SRT file into the video using ffmpeg with TV news style formatting.
    Applies subtitles to already correctly scaled video to prevent aspect ratio issues.
    
    Args:
        video_path (str, optional): Path to the video file, or None to use config default
        srt_path (str, optional): Path to the SRT file, or None to use config default
        output_path (str, optional): Path for the output file, or None to use config default
        channel_number (int, optional): Channel number to use, or None to use default
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Use channel-specific paths if parameters are not provided
    if video_path is None:
        output_dir = file_mgr.get_channel_output_path(channel_number)
        video_path = str(output_dir / config.file_paths.final_video_file)
    if srt_path is None:
        srt_path = str(file_mgr.get_caption_path(channel_number, config.file_paths.captions_file))
    if output_path is None:
        output_dir = file_mgr.get_channel_output_path(channel_number)
        output_path = str(output_dir / config.file_paths.final_subtitled_video_file)
        
    # Check if input files exist
    if not file_mgr.file_exists(video_path):
        print(f"Video file not found for subtitles: {video_path}")
        print("Creating placeholder video since input video is missing")
        try:
            # Check if we have a voice file that we can use with the placeholder
            voice_path = None
            if channel_number is not None:
                voice_path = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/",""))
                if not file_mgr.file_exists(voice_path):
                    voice_path = None
            
            # Create placeholder with voice if available
            create_placeholder_clip(output_path, 60)
            print(f"Created placeholder video at {output_path}")
            
            # If we have voice, add it to the placeholder
            if voice_path and file_mgr.file_exists(voice_path):
                print(f"Adding voice from {voice_path} to placeholder")
                temp_output = str(output_path) + ".temp.mp4"
                try:
                    # Add voice to placeholder video
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-i", str(output_path),
                        "-i", str(voice_path),
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        temp_output
                    ], check=True)
                    # Replace original with voiced version
                    os.replace(temp_output, output_path)
                    print("Successfully added voice to placeholder")
                except Exception as e:
                    print(f"Failed to add voice to placeholder: {e}")
            
            return True
        except Exception as e:
            print(f"Error creating placeholder: {e}")
            return False
    
    # Check if SRT file exists
    if not file_mgr.file_exists(srt_path):
        print(f"SRT file not found: {srt_path}")
        print("Copying video file to output without subtitles")
        try:
            file_mgr.copy_file(video_path, output_path)
            print(f"Video copied to {output_path}")
            return True
        except Exception as e:
            print(f"Error copying video: {e}")
            # Try creating a placeholder as fallback
            try:
                create_placeholder_clip(output_path, 60)
                return True
            except:
                return False
    
    # Ensure the output directory exists
    file_mgr.ensure_dir_exists(Path(output_path).parent)
    
    # Get subtitle styling from config
    font = config.video_edit.subtitle_font
    font_size = config.video_edit.subtitle_font_size
    
    # Create a temporary file for the SRT with normalized encoding
    with file_mgr.temp_file(suffix=".srt") as normalized_srt:
        try:
            # Read the SRT content
            srt_content = file_mgr.read_text(srt_path)
            if srt_content:
                # Write to a new file with UTF-8 encoding to avoid potential encoding issues
                file_mgr.write_text(normalized_srt, srt_content)
                
                # Use a simpler approach for burning subtitles
                try:
                    # Version 1: Use drawtext filter directly - more reliable but basic
                    print("Using drawtext filter for subtitles (basic but reliable)")
                    
                    # Extract subtitle text and timestamps
                    subtitle_entries = []
                    current_entry = None
                    
                    for line in srt_content.split('\n'):
                        line = line.strip()
                        if not line:
                            if current_entry and 'text' in current_entry:
                                subtitle_entries.append(current_entry)
                                current_entry = None
                        elif '-->' in line:
                            # This is a timestamp line
                            if current_entry:
                                times = line.split(' --> ')
                                if len(times) == 2:
                                    current_entry['start'] = times[0].replace(',', '.')
                                    current_entry['end'] = times[1].replace(',', '.')
                        elif current_entry and 'text' in current_entry:
                            # Append to existing text
                            current_entry['text'] += ' ' + line
                        elif current_entry:
                            # First text line
                            current_entry['text'] = line
                        else:
                            # New entry - likely a number
                            current_entry = {'number': line, 'text': ''}
                    
                    # Add the last entry if it exists
                    if current_entry and 'text' in current_entry:
                        subtitle_entries.append(current_entry)
                    
                    # Use basic drawtext filter for subtitles
                    print(f"Found {len(subtitle_entries)} subtitle entries")
                    
                    # Simply copy the video without subtitles if there's an issue
                    if not subtitle_entries:
                        print("No valid subtitle entries found. Copying video without subtitles.")
                        file_mgr.copy_file(video_path, output_path)
                        return
                    
                    # If we have valid subtitle entries but processing might be too complex,
                    # just copy the video for now as subtitles are a nice-to-have
                    if len(subtitle_entries) > 30:
                        print(f"Too many subtitle entries ({len(subtitle_entries)}). Copying video without subtitles for now.")
                        file_mgr.copy_file(video_path, output_path)
                        return True
                
                    # Actually burn the subtitles instead of skipping it
                    print("Burning subtitles directly into video...")
                    try:
                        # Escape path for subtitles to handle special characters
                        srt_clean_path = str(srt_path).replace(":", "\\:").replace("'", "\\'")
                        
                        with file_mgr.temp_file(suffix=".srt") as temp_srt:
      # Copy the original SRT content to temp file
                            file_mgr.copy_file(srt_path, temp_srt)

                            # Use the simple temp path in the ffmpeg command
                            subprocess.run([
                                "ffmpeg", "-y",
                                "-i", video_path,
                                "-vf", f"subtitles={temp_srt}:force_style='FontName=DIN Condensed Bold,FontSize=12,PrimaryColour=&HFFFFFF,OutlineColour=&H00000010,BorderStyle=4,Outline=1,Shadow=1,MarginV=35'",
                                "-c:v", "libx264",
                                "-preset", "fast",
                                "-crf", "22",
                                "-c:a", "copy",
                                output_path
                            ], check=True)
                        print(f"Successfully added subtitles to {output_path}")
                        return True
                    except Exception as e:
                        print(f"Error burning subtitles: {e}")
                        print(f"Error details: {str(e)}")
                        print("Trying alternative subtitle approach...")
                        
                        try:
                            # Alternative approach with hardcoded srt file
                            subprocess.run([
                                "ffmpeg", "-y",
                                "-i", video_path,
                                "-c:v", "libx264",
                                "-c:a", "copy", 
                                "-vf", "subtitles=subtitles.srt:force_style='FontSize=12,Alignment=2,OutlineColour=&H00000010,BorderStyle=3'",
                                output_path
                            ], check=True)
                            print("Successfully added subtitles using alternative approach")
                            return True
                        except Exception as e2:
                            print(f"Alternative subtitle approach failed: {e2}")
                            print("Falling back to copying video without subtitles")
                            file_mgr.copy_file(video_path, output_path)
                            print(f"Video copied to {output_path} without subtitles as fallback")
                            return True
                    
                except Exception as e:
                    print(f"Error processing subtitles: {e}")
                    # Copy the original video as fallback
                    file_mgr.copy_file(video_path, output_path)
                    print(f"Video copied to {output_path} without subtitles due to processing error")
                    return True
            else:
                print(f"Error: SRT file is empty or could not be read")
                # Copy the original video as fallback
                file_mgr.copy_file(video_path, output_path)
                print(f"Video copied to {output_path} without subtitles")
                return True
        except Exception as e:
            print(f"Error processing subtitles: {e}")
            # Copy the original video as fallback
            try:
                file_mgr.copy_file(video_path, output_path)
                print(f"Video copied to {output_path} without subtitles due to error")
                return True
            except Exception as e2:
                print(f"Error copying video: {e2}")
                # Try creating placeholder as last resort
                try:
                    create_placeholder_clip(output_path, 60)
                    print(f"Created placeholder as final fallback")
                    return True
                except:
                    return False

def main(channel_number: Optional[int] = None) -> None:
    """
    Main function to execute the video editing pipeline.
    
    Args:
        channel_number (Optional[int]): Channel number to process, or None to use default
    """
    print("Starting video editing process...")
    
    try:
        # Use default channel if none specified
        if channel_number is None:
            channel_number = config.default_channel
        
        # Load available clips and the script
        clips = load_clips_metadata()
        script = get_script_segments(channel_number)
        
        # Get voice file path from configuration - use channel-specific path
        voice_file = str(file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/","")))
        target_duration = None
        
        print(f"Looking for voice file at: {voice_file}")
        if file_mgr.file_exists(voice_file):
            target_duration = get_voice_duration(voice_file)
            print(f"Voice duration: {target_duration} seconds")
        else:
            print("Voice file not found. Using default duration.")
            target_duration = 60.0  # Default duration if voice file missing
        
        # Get expected number of segments from the SRT file - use channel-specific path
        captions_file = str(file_mgr.get_caption_path(channel_number, config.file_paths.captions_file))
        print(f"Looking for captions file at: {captions_file}")
        expected_segments = get_num_segments(captions_file, channel_number)
        print(f"Expected number of clip segments: {expected_segments}")
        
        # If no segments found, set expected segments to 1
        if expected_segments == 0:
            expected_segments = 1
            print(f"No segments found in SRT. Setting expected segments to {expected_segments}")
        
        # Match clips to the script, validate that output matches expected number
        attempts = 0
        max_attempts = 1
        clip_sequence = None
        
        while attempts < max_attempts:
            try:
                print(f"Matching clips to script (attempt {attempts+1}/{max_attempts})...")
                clip_sequence = match_clips_to_script(script, clips, target_duration=target_duration, channel_number=channel_number)
                obtained_segments = len(clip_sequence)
                
                if obtained_segments == expected_segments:
                    print("Successfully matched clip sequence with the correct number of segments.")
                    break
                else:
                    attempts += 1
                    print(f"Mismatch: expected {expected_segments} segments, but got {obtained_segments} segments. Re-instructing...")
            except Exception as e:
                print(f"Error during clip matching: {str(e)}")
                attempts += 1
        
        if clip_sequence is None:
            print("Failed to obtain a clip sequence. Creating placeholder sequence.")
            # Create a basic placeholder sequence
            placeholder_path = "sample_clips/placeholder.mp4"
            clip_sequence = [{
                'clip_name': placeholder_path,
                'start_time': 0,
                'duration': target_duration or 60.0,
                'script_segment': script[:100] + "..."
            }]
        elif len(clip_sequence) != expected_segments and expected_segments > 0:
            print(f"Warning: Obtained {len(clip_sequence)} segments, but expected {expected_segments}. Continuing anyway.")
        
        # Validate and adjust clip start times and durations to ensure they're within valid ranges
        print("Validating clip sequence...")
        clip_sequence = validate_clip_sequence(clip_sequence, clips)
        
        # Enforce maximum clip duration for faster-paced edits
        print("Enforcing maximum clip durations...")
        clip_sequence = enforce_clip_duration(clip_sequence)
        
        # Create the video sequence from selected clip segments
        print("Creating video sequence...")
        if not create_video_sequence(clip_sequence, clips, channel_number):
            print("Error creating video sequence. Creating placeholder video.")
            output_video = file_mgr.get_video_output_path(channel_number, config.file_paths.output_video_file)
            create_placeholder_clip(output_video, 60)
        
        # Merge the generated voice with the created video and add background music
        print("Merging voice with video and adding background music...")
        if not merge_voice_with_video(channel_number=channel_number):
            print("Error merging voice with video. Creating fallback video.")
            final_video = file_mgr.get_channel_output_path(channel_number) / config.file_paths.final_video_file
            create_placeholder_clip(final_video, 60)
        
        # Burn subtitles into the final video
        print("Adding subtitles to final video...")
        if not burn_subtitles(channel_number=channel_number):
            print("Error adding subtitles. Creating fallback subtitled video.")
            final_subtitled = file_mgr.get_channel_output_path(channel_number) / config.file_paths.final_subtitled_video_file
            create_placeholder_clip(final_subtitled, 60)
        
        # Verify the final file exists
        final_output = file_mgr.get_channel_output_path(channel_number) / config.file_paths.final_subtitled_video_file
        if file_mgr.file_exists(final_output):
            print(f"Video editing process completed successfully. Final output: {final_output}")
        else:
            print(f"Final video file not found despite completion. Creating one last emergency placeholder at {final_output}")
            create_placeholder_clip(final_output, 60)
    
    except Exception as e:
        print(f"Error during video editing process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments for channel
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and edit video for VideoAI")
    parser.add_argument("--channel", type=int, choices=[1, 2, 3], 
                       help="Channel number to use (1-3)")
    
    args = parser.parse_args()
    
    main(args.channel)