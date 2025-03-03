import os
import json
import requests
from dotenv import load_dotenv
import random

def generate_voice(script_text, channel_number=1):
    """
    Converts the provided `script_text` into speech using the updated ElevenLabs text-to-speech API.
    
    Args:
        script_text (str): The text to convert to speech
        channel_number (int): The channel number (1, 2, or 3) to determine which voice to use
    """
    # Load environment variables
    load_dotenv(override=True)
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY is not set in the environment.")
        return

    # Map channels to specific voice IDs
    voice_mapping = {
        1: "voice_id1",  # Channel 1
        2: "voice_id2",  # Channel 2 
        3: "voice_id3"   # Channel 3
    }
    
    # Get the voice ID for the specified channel
    voice_id = voice_mapping[channel_number]
    
    # Build the endpoint URL with the output_format query parameter.
    url = f"https://api.elevenlabs.io/v1/text-to-speech/29vD33N1CtxCmqQRPOHJ?output_format=mp3_44100_128"

    payload = {
        "text": script_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.35,  # Controls variability (0-1)
            "similarity_boost": 0.55,  # Controls how closely output matches voice (0-1)
            "style": 0.1,  # Controls expressiveness (0-1)
            "use_speaker_boost": True  # Enhances voice clarity
        }
    }

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    # Ensure output directory exists
    output_dir = r"D:\videoai\output_voice"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_voice.mp3")

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Write the binary audio content to file.
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Voice file saved to {output_file}")
    except requests.RequestException as e:
        print("Request failed:", e)

def main(channel_number=1):
    script_file_path = "generated_script.txt"
    try:
        with open(script_file_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    except Exception as e:
        print(f"Failed to load script file {script_file_path}: {e}")
        return

    generate_voice(script_text, channel_number)

if __name__ == "__main__":
    main()
