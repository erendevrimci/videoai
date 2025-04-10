from typing import Optional
from pydantic import BaseModel

class VoiceoverRequest(BaseModel):
    script_id: int
    language: Optional[str] = None
    voice: Optional[str] = None
    channel_number: int



