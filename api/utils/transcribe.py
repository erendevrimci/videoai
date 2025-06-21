import os
from openai import OpenAI
import tempfile
import base64

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Transcribes the given audio bytes using OpenAI's Whisper API and returns the text.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        return None
        
    client = OpenAI(api_key=api_key)
    
    try:
        # Whisper API expects a file-like object with a name. We use a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file.seek(0)
            temp_file_name = temp_audio_file.name

        with open(temp_file_name, "rb") as file_to_transcribe:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_to_transcribe,
                response_format="text"
            )
        
        os.remove(temp_file_name)
            
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        if 'temp_file_name' in locals() and os.path.exists(temp_file_name):
            os.remove(temp_file_name)
        # In case of an API error, it's good to log it.
        # Returning None to indicate failure.
        return None 