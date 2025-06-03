from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class WordTiming(BaseModel):
    word: str
    start: float  # Saniye cinsinden
    end: float    # Saniye cinsinden

class CaptionSegment(BaseModel):
    id: str
    text: str
    start: float
    end: float
    words: List[WordTiming]

class CaptionDetails(BaseModel):
    id: int
    project_id: int
    voice_over_id: int
    caption_file: str
    segments: List[CaptionSegment]
    total_duration: float
    created_at: str

class CaptionResponse(BaseModel):
    success: bool
    message: str
    id: Optional[int] = None
    caption: Optional[CaptionDetails] = None

class CaptionListResponse(BaseModel):
    success: bool
    message: str
    captions: Optional[List[CaptionDetails]] = None


