#!/usr/bin/env python3
import write_script
import voice_over
import video_edit
import write_title_desc
import upload_video
from captions import generate_subtitles
import sys
import os
import time

def process_channel(channel_number):
    print(f"\nStarting process for Channel {channel_number}...\n")

    # 1. Generate the YouTube script.
    print("--- Step 1: Generating Script ---")
    write_script.main()
    
    # 2. Generate the voice-over using ElevenLabs.
    print("\n--- Step 2: Generating Voice-over ---")
    voice_over.main(channel_number)
    
    # 3. Generate captions (SRT file) using OpenAI's Whisper API.
    print("\n--- Step 3: Generating Captions ---")
    generate_subtitles("output_voice/generated_voice.mp3", "generated_voice.srt")

    # 4. Edit and assemble the video.
    print("\n--- Step 4: Editing Video ---")
    video_edit.main()
    
    # 5. Generate the title and description.
    print("\n--- Step 5: Generating Title & Description ---")
    write_title_desc.main()

    # 6. Upload the video to YouTube for specific channel
    print(f"\n--- Step 6: Uploading Video to YouTube Channel {channel_number} ---")
    if channel_number == 1:
        upload_video.upload_to_channel1()
    elif channel_number == 2:
        upload_video.upload_to_channel2()
    else:
        upload_video.upload_to_channel3()

    # Define the target filenames
    target_files = {
        "final_output_with_subtitles.mp4": f"final_output_channel{channel_number}.mp4",
        "generated_script.txt": f"script_channel{channel_number}.txt", 
        "youtube_info.json": f"youtube_info_channel{channel_number}.json"
    }

    # Rename files with error handling
    for source, target in target_files.items():
        try:
            # Remove target file if it already exists
            if os.path.exists(target):
                os.remove(target)
            # Perform the rename
            if os.path.exists(source):
                os.rename(source, target)
        except OSError as e:
            print(f"Warning: Could not rename {source} to {target}: {e}")

def main():
    print("Starting the full automated process for all channels...\n")

    # Process each channel sequentially
    for channel_number in range(1, 4):
        process_channel(channel_number)
        
        # Add a delay between channels to avoid rate limits
        if channel_number < 3:
            print(f"\nWaiting 60 seconds before processing next channel...")
            time.sleep(60)

    print("\nAll channels processed successfully!")

if __name__ == "__main__":
    main()
