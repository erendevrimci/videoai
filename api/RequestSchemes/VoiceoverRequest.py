from typing import Optional
from pydantic import BaseModel

class VoiceoverRequest(BaseModel):
    project_id: int
    script_id: int
    voice_id: Optional[str] = "9BWtsMINqrJLrRacOk9x"
    language: Optional[str] = None
    voice: Optional[str] = None
    channel_number: Optional[int] = 1
    similarity_boost: Optional[float] = 0.5
    stability: Optional[float] = 0.5



