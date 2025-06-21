from supabase import create_client
import os
from mutagen.mp3 import MP3
import io
import base64
from pydantic import BaseModel

class UploadVoiceoverResponse(BaseModel):
    id: int
    voice_name: str
    duration: int
    project_id: int
    user_id: str
    created_at: str

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

def insert_voiceover_to_db(user_id: str, project_id: int, voice_over_name: str, voice_over_duration: int):
    insert_result = supabase.table("voice_over").insert({
        "user_id": user_id,
        "project_id": project_id,
        "voice_name": voice_over_name,
        "duration": voice_over_duration,
    }).execute()
    return insert_result.data[0]

def upload_voiceover_to_storage(user_id: str, project_id: int, audio_file_base64: str, voice_over_name: str)->UploadVoiceoverResponse:
    try:
        audio_file_bytes = base64.b64decode(audio_file_base64)
        file_like_object = io.BytesIO(audio_file_bytes)
        audio = MP3(file_like_object)
        duration = int(audio.info.length)
        storage_insert = supabase.storage.from_("voice-over-files").upload(
            path=voice_over_name,
            file=audio_file_bytes,
            file_options={"content-type": "audio/mpeg"}
        )
        
        return insert_voiceover_to_db(user_id, project_id, voice_over_name, duration)
        
    except Exception as e:
        return None
        


