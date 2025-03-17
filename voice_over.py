import json
import requests
import sys
from typing import Optional

# Import the config module and file manager
from config import config, get_channel_config
from file_manager import FileManager

def generate_voice(script_text: str, channel_number: int = 1) -> Optional[str]:
    """
    Converts the provided script text into speech using the ElevenLabs text-to-speech API.
    
    Args:
        script_text (str): The text to convert to speech
        channel_number (int): The channel number to determine which voice to use
        
    Returns:
        Optional[str]: Path to the generated voice file, or None if generation failed
    """
    # Initialize file manager
    file_mgr = FileManager()
    
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
    
    # Define output file paths using the file manager
    channel_voice_file = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/","").replace(".mp3",""))
    
    
    
    try:
        print(f"Generating voice using ElevenLabs API (voice ID: {voice_id})...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Write the binary audio content to both file locations using the file manager
        file_mgr.write_binary(channel_voice_file, response.content)
        
        
        print(f"Voice files saved to:")
        print(f"  - {channel_voice_file}")
        
        return str(channel_voice_file)
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
    # Initialize file manager
    file_mgr = FileManager()
    
    # Use default channel if none specified
    if channel_number is None:
        channel_number = config.default_channel
    
    # Try multiple script file path options
    script_file_paths = [
        file_mgr.get_script_path(channel_number, config.file_paths.script_file),
        file_mgr.get_abs_path(config.file_paths.script_file)
    ]
    
    script_text = None
    used_path = None
    
    # Try each path until we find one that works
    for path in script_file_paths:
        script_text = file_mgr.read_text(path)
        if script_text is not None:
            used_path = path
            break
    
    if script_text is None:
        # If all paths failed, try generating a script first
        print(f"Failed to load script file from any path: {script_file_paths}")
        print("Attempting to generate a script first...")
        
        import write_script
        write_script.main(channel_number)
        
        # Try again after script generation
        for path in script_file_paths:
            script_text = file_mgr.read_text(path)
            if script_text is not None:
                used_path = path
                break
                
        if script_text is None:
            print("Could not generate or load a script. Aborting voice generation.")
            return
    
    print(f"Reading script from {used_path}...")
    
    # Generate voice
    voice_file = generate_voice(script_text, channel_number)
    
    if voice_file:
        print(f"Voice generation completed successfully.")
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
