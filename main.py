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
from typing import List, Optional

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
from config import config, get_channel_config
from file_manager import FileManager

# Initialize the file manager
file_mgr = FileManager()

# Create a module-specific logger
if has_logging:
    logger = Logger.get_logger("main")

def process_channel(channel_number: int, steps: List[str] = None) -> None:
    """
    Process a single channel through the complete video pipeline.
    
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
            
            # Use channel-specific paths for voice and captions
            voice_file = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/",""))
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
        logger.info(f"Arguments: channel={args.channel}, steps={args.steps}, delay={args.delay}")
    
    # Process specific channel or all channels
    if args.channel:
        if has_logging:
            logger.info(f"Processing channel {args.channel}")
        process_channel(args.channel, steps)
    else:
        start_msg = "Starting the full automated process for all channels..."
        if has_logging:
            logger.info(start_msg)
        else:
            print(f"{start_msg}\n")
        
        # Process each channel sequentially
        for channel_number in range(1, 4):
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
