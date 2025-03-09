"""
Example of using the FileManager class with the voice_over module.

This example demonstrates how to refactor the voice_over.py module to use
the FileManager class for file operations.
"""
import sys
import requests
from typing import Optional
from pathlib import Path

# Import the config module and FileManager
from config import config, get_channel_config
from file_manager import FileManager

# Initialize file manager
file_mgr = FileManager()

def generate_voice(script_text: str, channel_number: int = 1) -> Optional[Path]:
    """
    Converts the provided script text into speech using the ElevenLabs text-to-speech API.
    
    Args:
        script_text (str): The text to convert to speech
        channel_number (int): The channel number to determine which voice to use
        
    Returns:
        Optional[Path]: Path to the generated voice file, or None if generation failed
    """
    # Get API key from configuration
    api_key = config.elevenlabs.api_key
    if not api_key:
        print("Error: ElevenLabs API key is not set in the configuration")
        return None
    
    # Get channel-specific configuration
    try:
        channel_config = get_channel_config(channel_number)
        voice_id = channel_config.voice_id
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Using default voice ID: {config.elevenlabs.default_voice_id}")
        voice_id = config.elevenlabs.default_voice_id
    
    # Get voice settings from configuration
    voice_settings = {
        "stability": config.elevenlabs.stability,
        "similarity_boost": config.elevenlabs.similarity_boost,
        "style": config.elevenlabs.style,
        "use_speaker_boost": config.elevenlabs.use_speaker_boost
    }
    
    # Build the endpoint URL with the output_format query parameter
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
    
    # Prepare API request
    payload = {
        "text": script_text,
        "model_id": config.elevenlabs.model_id,
        "voice_settings": voice_settings
    }
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Get output file paths using FileManager
    # These methods automatically ensure directories exist
    channel_voice_file = file_mgr.get_audio_output_path(channel_number, "generated_voice")
    legacy_voice_file = file_mgr.get_abs_path("voice/generated_voice.mp3")
    
    # Ensure legacy directory exists (channel directory is automatically created)
    file_mgr.ensure_dir_exists(legacy_voice_file.parent)
    
    try:
        print(f"Generating voice using ElevenLabs API (voice ID: {voice_id})...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Write the binary audio content to both file locations using FileManager
        file_mgr.write_binary(channel_voice_file, response.content)
        file_mgr.write_binary(legacy_voice_file, response.content)
        
        print(f"Voice files saved to:")
        print(f"  - {channel_voice_file}")
        print(f"  - {legacy_voice_file}")
        
        return channel_voice_file
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e.response, 'status_code') and hasattr(e.response, 'text'):
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return None

def main(channel_number: Optional[int] = None) -> None:
    """
    Main function to generate voice from script.
    
    Args:
        channel_number (Optional[int]): Channel number to use. If None, uses default channel.
    """
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Get script file path using FileManager
    script_file_path = file_mgr.get_script_path(channel_number)
    
    # Read script using FileManager with built-in error handling
    script_text = file_mgr.read_text(script_file_path)
    if script_text is None:
        print(f"Failed to load script file {script_file_path}")
        return
    
    print(f"Read script from {script_file_path} ({len(script_text)} characters)")
    
    # Generate voice
    voice_file = generate_voice(script_text, channel_number)
    
    if voice_file:
        print(f"Voice generation completed successfully: {voice_file}")
    else:
        print("Voice generation failed.")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            channel_number = int(sys.argv[1])
            main(channel_number)
        except ValueError:
            print(f"Error: Invalid channel number '{sys.argv[1]}'. Using default channel.")
            main()
    else:
        main()