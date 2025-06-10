from pydantic import BaseModel
from typing import List, Optional

class TimelineSegment(BaseModel):
    prompt: str
    cfg_scale: Optional[float] = None
    duration: int
    start_image_id: str
    end_image_id: Optional[str] = None
    start_image: str
    end_image: Optional[str] = None
    
    

class GenerateTimelineVideoRequest(BaseModel):
    model: str
    segments: List[TimelineSegment]
    aspect_ratio: str
    


