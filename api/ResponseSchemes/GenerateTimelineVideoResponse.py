from pydantic import BaseModel
from typing import Optional, List
from ..RequestSchemes.GenerateTimelineVideoRequest import TimelineSegment



class GenerateTimelineVideoResponse(BaseModel):
    success: bool
    message: str
    segments: Optional[List[TimelineSegment]] = None
    video_urls: Optional[List[str]] = None
