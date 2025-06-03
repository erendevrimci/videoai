from typing import Optional, List
from pydantic import BaseModel


class VoiceoverHistory(BaseModel):
    id: int
    name: str
    duration: int
    created_at: str
    url: Optional[str] = None

class VoiceoverResponse(BaseModel):
    success: bool
    message: str
    voice_over_url: Optional[str] = None
    voice_over_history: Optional[List[Optional[VoiceoverHistory]]] = None
