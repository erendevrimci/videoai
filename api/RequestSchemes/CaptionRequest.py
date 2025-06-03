from pydantic import BaseModel
from typing import Optional, List

class WordTimingUpdate(BaseModel):
    word: str
    start: float
    end: float

class CaptionSegmentUpdate(BaseModel):
    id: str
    text: str
    start: float
    end: float
    words: List[WordTimingUpdate]

class CaptionRequest(BaseModel):
    project_id: int
    voice_over_id: int
    channel_number: Optional[int] = 1
    use_timeline: Optional[bool] = False

class UpdateCaptionSegmentsRequest(BaseModel):
    segments: List[CaptionSegmentUpdate]

