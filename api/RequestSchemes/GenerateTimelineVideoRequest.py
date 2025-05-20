from pydantic import BaseModel
from typing import List

class TimelineSegment(BaseModel):
    prompt: str
    cfg_scale: float
    duration: int
    start_image_id: int
    end_image_id: int
    start_image: str
    end_image: str
    
    

class GenerateTimelineVideoRequest(BaseModel):
    model: str
    segments: List[TimelineSegment]
    aspect_ratio: str
    


