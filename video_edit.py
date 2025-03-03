import os
from openai import OpenAI
from pathlib import Path
import json
import subprocess
from typing import List, Dict
from dotenv import load_dotenv
import random

load_dotenv(override=True)

def load_clips_metadata() -> List[Dict]:
    """Load and parse the clips metadata from clips_label.md"""

    clips = []
    with open('clips/clips_label.md', 'r') as f:
        content = f.read().split('------------------------------------------------------------------------')
        
    for clip_info in content:
        if not clip_info.strip():
            continue
        
        lines = clip_info.strip().split('\n')
        clip = {}
        for line in lines:
            if line.startswith('Clip Name:'):
                clip['name'] = line.replace('Clip Name:', '').strip()
            elif line.startswith('Clip Description:'):
                clip['description'] = line.replace('Clip Description:', '').strip()
            elif line.startswith('Clip Notes:'):
                clip['notes'] = line.replace('Clip Notes:', '').strip()
            elif line.startswith('Clip Length:'):
                time_str = line.replace('Clip Length:', '').strip()
                # Handle HH:MM:SS or MM:SS format
                time_parts = time_str.split(':')
                if len(time_parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(int, time_parts)
                    clip['duration'] = hours * 3600 + minutes * 60 + seconds
                else:  # MM:SS
                    minutes, seconds = map(int, time_parts)
                    clip['duration'] = minutes * 60 + seconds
        
        if clip:
            clips.append(clip)
    
    return clips

def get_script_segments() -> str:
    """Load the script content"""
    with open('generated_script.txt', 'r', encoding='utf-8') as f:
        return f.read()

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

def get_num_segments(srt_file: str = "generated_voice.srt") -> int:
    """Determine the number of subtitle segments in the SRT file"""
    with open(srt_file, "r", encoding="utf-8") as f:
        srt_content = f.read().strip()
    subtitles = [seg for seg in srt_content.split('\n\n') if seg.strip() != '']
    return len(subtitles)

def match_clips_to_script(script: str, clips: List[Dict], target_duration: float = None) -> List[Dict]:
    """Use OpenAI to match clips to script segments based on SRT timestamps and content."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a set of available clip names for validation
    available_clips = {clip['name'] for clip in clips}
    
    # Read SRT file for timing information
    with open('generated_voice.srt', 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Parse SRT to get actual segment durations
    srt_segments = [seg.strip() for seg in srt_content.split('\n\n') if seg.strip()]
    segment_timings = []
    
    for segment in srt_segments:
        lines = segment.split('\n')
        if len(lines) >= 2:
            times = lines[1].split(' --> ')
            if len(times) == 2:
                start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(times[0].replace(',', '.').split(':'))))
                end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(times[1].replace(',', '.').split(':'))))
                duration = end - start
                segment_timings.append(duration)
    
    # Modify the clips info to include ONLY available clips
    clips_info = "\n".join([
        f"Clip: {c['name']}\nDescription: {c['description']}\nNotes: {c['notes']}\n"
        f"Duration: {c['duration']}s\n"
        f"Possible start times: 0 to {max(0, c['duration'] - 10)} seconds"
        for c in clips
    ])
    
    target_duration_text = ""
    if target_duration is not None:
        target_duration_text = f"\nTotal Generated Voice Duration: {int(target_duration)} seconds."
    
    prompt = f"""Given these available video clips along with their metadata:
    
{clips_info}
{target_duration_text}

And this SRT file with timestamps and script segments:

{srt_content}

Your task is to create a sequence of clips that best matches the voiceover content and timing. When selecting clip segments:
- Use the EXACT SRT timestamps to ensure clips align with the voiceover timing
- Each clip segment MUST match the exact duration of its corresponding SRT segment
- Use the Clip Description and Clip Notes to determine which clips best match each part of the script
- For each clip selection, choose a random start time within the possible start times range
- You may use the same clip multiple times by choosing different start times if needed
- In the first 30 seconds, we MUST ensure that at least 3 different clips are used
- For longer clips (1 minute), make sure to utilize different parts of the clip by selecting varied start times

Return the response as a JSON array where each object has:
- clip_name: the filename of the clip to use
- start_time: when to start using the clip (in seconds from the clip's beginning)
- duration: length of the clip segment (MUST match the SRT segment duration)
- script_segment: the part of the script that this clip should align with

Format the response as valid JSON only, no additional text."""
    
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "developer", 
                "content": "You are a video editing assistant with expertise in aligning clip metadata with script content and timing for fast-paced, dynamic edits."
            },
            {"role": "user", "content": prompt}
        ]
    )
    
    try:
        clip_sequence = json.loads(response.choices[0].message.content)
        # Validate clips and filter out any that don't exist
        validated_sequence = []
        for segment in clip_sequence:
            if segment['clip_name'] in available_clips:
                validated_sequence.append(segment)
            else:
                print(f"Warning: Skipping non-existent clip '{segment['clip_name']}'")
                # You might want to retry or use a fallback clip here
        
        # Ensure the durations match the SRT segments
        for i, segment in enumerate(validated_sequence):
            if i < len(segment_timings):
                segment['duration'] = segment_timings[i]
        
        return validated_sequence
    except json.JSONDecodeError:
        raise Exception("Failed to get valid JSON response from OpenAI")

def enforce_clip_duration(clip_sequence: List[Dict]) -> List[Dict]:
    """Ensure that every clip segment's duration does not exceed 10 seconds"""
    for clip in clip_sequence:
        if clip['duration'] > 10:
            print(f"Adjusting clip '{clip['clip_name']}' duration from {clip['duration']}s to 10s for faster-paced edits.")
            clip['duration'] = 10
    return clip_sequence

def validate_clip_sequence(clip_sequence: List[Dict], clips_metadata: List[Dict]) -> List[Dict]:
    """Validate and adjust clip start times and durations to ensure they're within valid ranges."""
    clips_dict = {clip['name']: clip['duration'] for clip in clips_metadata}
    used_segments = {}  # clip_name -> list of (start_time, end_time) tuples
    
    print(f"Total segments to process: {len(clip_sequence)}")
    total_duration_before = sum(segment['duration'] for segment in clip_sequence)
    print(f"Total duration before validation: {total_duration_before:.2f} seconds")
    
    for i, segment in enumerate(clip_sequence):
        clip_name = segment['clip_name']
        total_duration = clips_dict.get(clip_name, 0)
        original_duration = segment['duration']  # Store the original duration
        
        print(f"\nProcessing segment {i+1}/{len(clip_sequence)}")
        print(f"Clip: {clip_name}, Original Duration: {original_duration:.2f}s")
        
        if clip_name not in used_segments:
            used_segments[clip_name] = []
        
        # For clips longer than 30 seconds, try to find an unused segment
        if total_duration > 30:
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

def create_video_sequence(clip_sequence: List[Dict]) -> None:
    """Use ffmpeg to concatenate the selected clip segments into a video (output_video.mp4)"""
    # Create a temporary file listing all clips to concatenate with precise trim parameters.
    with open('temp_list.txt', 'w') as f:
        for clip in clip_sequence:
            clip_path = os.path.join('clips', clip['clip_name'])
            start_time = clip.get('start_time', 0)
            duration = clip['duration']
            f.write(f"file '{clip_path}'\n")
            f.write(f"inpoint {start_time}\n")
            f.write(f"outpoint {start_time + duration}\n")
    
    # Use ffmpeg to concatenate the clips using the inpoint/outpoint method for accurate trimming.
    subprocess.run([
        'ffmpeg',
        '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'temp_list.txt',
        '-c', 'copy',
        'output_video.mp4'
    ], check=True)
    
    # Clean up temporary file.
    os.remove('temp_list.txt')

def merge_voice_with_video(video_path: str, voice_path: str, output_path: str) -> None:
    """Merge the generated voice audio (MP3) with the video (MP4) and add background music as a second audio layer.
    Voice volume is set to 1.2 and background music volume to 0.2.
    """
    bgm_folder = "background_music"
    bgm_files = [os.path.join(bgm_folder, f) for f in os.listdir(bgm_folder) if f.lower().endswith(".mp3")]
    
    if bgm_files:
        bgm_file = random.choice(bgm_files)
        print(f"Selected background music: {bgm_file}")
        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", voice_path,
            "-stream_loop", "-1",
            "-i", bgm_file,
            "-filter_complex", 
            "[1:a]volume=1.4[voice];[2:a]volume=0.1[bgm];[voice][bgm]amix=inputs=2:duration=first[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
    else:
        print("No background music files found. Merging video with voice only.")
        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", voice_path,
            "-filter_complex", "[1:a]volume=1.4[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error merging video, voice, and background music:", e)

def burn_subtitles(video_path: str, srt_path: str, output_path: str) -> None:
    """Burn subtitles from the SRT file into the video using ffmpeg so that they appear at the bottom (TV news style)
    with a grey background behind the text.
    """
    # The force_style parameters:
    # - Alignment=2: Bottom-center alignment
    # - FontName=Arial: Clean, readable font
    # - FontSize=18: Matches the size in the example
    # - MarginV=10: Adds some vertical margin from the bottom
    # - BorderStyle=4: Enables background box
    # - BackColour=&H80000000: Semi-transparent black background
    # - PrimaryColour=&HFFFFFF: White text
    subtitles_filter = (
        f"subtitles={srt_path}:"
        "force_style='Alignment=2,FontName=Arial,FontSize=18,MarginV=10,"
        "BorderStyle=4,BackColour=&H80000000,PrimaryColour=&HFFFFFF'"
    )
    
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", subtitles_filter,
        "-c:a", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Final video with burned subtitles created: {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error burning subtitles:", e)

def main():
    # Load available clips and the script.
    clips = load_clips_metadata()
    script = get_script_segments()
    
    # Obtain the generated voice's duration, if available.
    voice_file = os.path.join("voice", "generated_voice.mp3")
    target_duration = None
    if os.path.exists(voice_file):
        target_duration = get_voice_duration(voice_file)
        print(f"Voice duration: {target_duration} seconds")
    else:
        print("Voice file not found. Proceeding without target voice duration.")
    
    # Get expected number of segments from the SRT file.
    expected_segments = get_num_segments("generated_voice.srt")
    print(f"Expected number of clip segments: {expected_segments}")
    
    # Match clips to the script, validate that output matches expected number.
    attempts = 0
    clip_sequence = None
    while attempts < 3:
        try:
            clip_sequence = match_clips_to_script(script, clips, target_duration=target_duration)
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

    if clip_sequence is None or len(clip_sequence) != expected_segments:
         print("Failed to obtain a valid clip sequence after multiple attempts. Exiting.")
         return
    
    # Validate and adjust clip start times and durations to ensure they're within valid ranges.
    clip_sequence = validate_clip_sequence(clip_sequence, clips)
    
    # Enforce that each clip segment's duration does not exceed 8 seconds for faster-paced edits.
    clip_sequence = enforce_clip_duration(clip_sequence)
    
    # Create the video sequence from selected clip segments.
    try:
        create_video_sequence(clip_sequence)
        print("Video sequence created: output_video.mp4")
    except Exception as e:
        print(f"Error during video sequence creation: {str(e)}")
        return
    
    # Merge the generated voice with the created video and add background music.
    try:
        merge_voice_with_video("output_video.mp4", voice_file, "final_output.mp4")
        print("Final video with merged voice and background music created: final_output.mp4")
    except Exception as e:
        print(f"Error merging voice, video, and background music: {str(e)}")
        return

    # Burn subtitles into the final video
    srt_file = "generated_voice.srt"  # Ensure the SRT file has been generated using your captions.py script
    try:
        burn_subtitles("final_output.mp4", srt_file, "final_output_with_subtitles.mp4")
    except Exception as e:
        print(f"Error burning subtitles: {e}")

if __name__ == "__main__":
    main()
