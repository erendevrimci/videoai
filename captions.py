"""
Captions Generation Module

This module handles the generation of subtitles (SRT format) for videos using OpenAI's Whisper API.
"""

import traceback
from pathlib import Path
from openai import OpenAI
from config import config
from file_manager import FileManager

# Initialize the file manager
file_mgr = FileManager()

def generate_subtitles(
    audio_file_path: str | Path, 
    output_srt_path: str | Path, 
    channel_number: int = None
) -> bool:
    """
    Generate subtitles (SRT file) from the given audio file using the OpenAI Whisper API.
    
    This function sends the generated voice mp3 file to the Whisper transcription API
    and requests the transcription in SRT format. The resulting subtitles are saved 
    into the output file.
    
    Args:
        audio_file_path: The path to the audio file (e.g., a generated_voice.mp3).
        output_srt_path: Path to save the SRT file.
        channel_number: Optional channel number to use for configuration.
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Generating subtitles from audio: {audio_file_path}")
    print(f"Output SRT path: {output_srt_path}")
    
    # Check if audio file exists
    if not Path(audio_file_path).exists():
        print(f"Error: Audio file not found at {audio_file_path}")
        return False
    
    # Initialize OpenAI client
    client = OpenAI(api_key=config.openai.api_key)
    
    try:
        # Convert paths to Path objects if they're strings
        audio_path = Path(audio_file_path) if isinstance(audio_file_path, str) else audio_file_path
        output_path = Path(output_srt_path) if isinstance(output_srt_path, str) else output_srt_path
        
        # Ensure the output directory exists using FileManager
        file_mgr.ensure_dir_exists(output_path.parent)
        
        print(f"Transcribing audio from {audio_path}...")
        # Read the audio file using FileManager and process with Whisper API
        audio_data = file_mgr.read_binary(audio_path)
        if audio_data is None:
            print(f"Error: Could not read audio file: {audio_path}")
            return False
        
        # Create a temporary file for the API to read since it expects a file object
        with file_mgr.temp_file(suffix=".mp3") as temp_audio_path:
            # Write the audio data to the temporary file
            file_mgr.write_binary(temp_audio_path, audio_data)
            
            print("Transcribing audio using OpenAI Whisper API...")
            with open(temp_audio_path, "rb") as audio_file:
                # Request transcription with SRT output format
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",  # Using hardcoded model as Whisper has limited models
                    file=audio_file,
                    response_format="srt"
                )
        
        # Write the SRT formatted text to the output file using FileManager
        success = file_mgr.write_text(output_path, transcription)
        if not success:
            print(f"Error: Could not write SRT file: {output_path}")
            return False
            
        # Also save a copy in the root directory for easier access by ffmpeg
        try:
            root_srt_path = Path('subtitles.srt')
            with open(root_srt_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            print(f"Also saved subtitles to root directory at {root_srt_path} for easier ffmpeg access")
        except Exception as e:
            print(f"Warning: Could not save copy of SRT to root directory: {e}")
            
        print(f"Subtitles successfully saved to {output_path}")
        return True
    except Exception as e:
        print(f"An error occurred during transcription: {str(e)}")
        print(traceback.format_exc())
        return False

def main(channel_number: int = None):
    """
    Main function to run the captions generation process.
    
    Args:
        channel_number: Optional channel number to use for configuration.
    """
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Get voice file path using FileManager
    audio_file_path = file_mgr.get_audio_output_path(channel_number, "generated_voice")
    
    # Get caption file path using FileManager
    output_srt_path = file_mgr.get_caption_path(channel_number)
    
    # Generate subtitles
    success = generate_subtitles(audio_file_path, output_srt_path, channel_number)
    
    if success:
        print(f"Caption generation completed successfully for channel {channel_number}")
    else:
        print(f"Caption generation failed for channel {channel_number}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate captions for video using OpenAI Whisper API")
    parser.add_argument("--channel", type=int, help="Channel number to use for configuration")
    
    args = parser.parse_args()
    
    main(channel_number=args.channel)
