from pydantic import BaseModel

class UploadVoiceoverRequest(BaseModel):
    project_id: int
    audio_file: str
    voice_over_name: str
