import json
import requests
import sys
from typing import Optional
import base64
# Import the config module and file manager
from config import config, get_channel_config
from file_manager import FileManager
from supabase import create_client
import os
from dotenv import load_dotenv
import uuid
import io
from mutagen.mp3 import MP3

load_dotenv()

def generate_voice(script_text: str,similarity_boost: float = 0.5, stability: float = 0.5, voice_id: str = "9BWtsMINqrJLrRacOk9x") -> Optional[bytes]:
    """
    Converts the provided script text into speech using the ElevenLabs text-to-speech API.
    
    Args:
        script_text (str): The text to convert to speech
        channel_number (int): The channel number to determine which voice to use
        
    Returns:
        Optional[str]: Path to the generated voice file, or None if generation failed
    """
    # Initialize file manager
    # file_mgr = FileManager()
    
    # Get API key from configuration
    api_key = config.elevenlabs.api_key
    if not api_key:
        print("Error: ElevenLabs API key is not set in the configuration")
        return None
    
    #
  
    
    # Get voice settings from configuration
    voice_settings = {
        "stability": stability,
        "similarity_boost": similarity_boost,
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
    # channel_voice_file = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/","").replace(".mp3",""))
    
    
    
    try:
        print(f"Generating voice using ElevenLabs API (voice ID: {voice_id})...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Write the binary audio content to both file locations using the file manager
        # file_mgr.write_binary(channel_voice_file, response.content)
        
        
        # print(f"Voice files saved to:")
        # print(f"  - {channel_voice_file}")
        
        return response.content
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e.response, 'status_code') and hasattr(e.response, 'text'):
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return None

def main(user_id: str,project_id: int,script_id: int,similarity_boost: float = 0.5, stability: float = 0.5,voice_id: Optional[str] = "9BWtsMINqrJLrRacOk9x") -> str:
    # """
    # Main function to generate voice from script.
    
    # Args:
    #     channel_number (Optional[int]): Channel number to use. If None, uses default channel.
    # """
   
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    
    script_query = supabase.table("scripts").select("script,topic").eq("project_id", project_id).eq("id", script_id).execute()
    
    script_text = script_query.data[0]["script"]
    script_name = script_query.data[0]["topic"]
    
    voice = generate_voice(script_text,similarity_boost, stability,voice_id)
    
    file_like_object = io.BytesIO(voice)
    audio = MP3(file_like_object)
    duration = int(audio.info.length)
    
    
    


    file_name = f"{script_name}_{uuid.uuid4()}.mp3"

    result = supabase.storage.from_("voice-over-files").upload(
        path=file_name,
        file=voice,
        file_options={"content-type": "audio/mpeg"}
    )

    signed_url_raw = supabase.storage.from_("voice-over-files").create_signed_url(file_name, 3600)
    
    signed_url = signed_url_raw.get('signedURL')
   

    response = supabase.table("voice_over").insert({
            "voice_name": file_name,
            "project_id": project_id,
            "duration": duration if duration else 0,
            "user_id": user_id
        }).execute()
    
    

    if "error" in response:
        raise Exception(f"Database Error: {response}")
    if voice:
        print(f"Voice generation completed successfully.")
    else:
        print("Voice generation failed.")
    
    return signed_url

if __name__ == "__main__":
    # Parse command line arguments
    main()
