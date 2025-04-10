from typing import Optional
from pydantic import BaseModel
import base64

class VoiceoverResponse(BaseModel):
    success: bool
    message: str
    voiceover: Optional[bytes] = None  # Base64 string olarak tutacağız
