from typing import Optional
from pydantic import BaseModel
import base64

class VoiceoverResponse(BaseModel):
    success: bool
    message: str
    voice_over_url: Optional[str] = None
