from pydantic import BaseModel
from typing import Optional
from typing import List

class Shot(BaseModel):
    id: int
    approved: bool
    video_url: Optional[str] = None
    duration: Optional[float] = None 
    suggestion: str
    explanation: str
    clip_name: Optional[str] = None
    script_segment: Optional[str] = None
    start_time: Optional[float] = None
    shot_index: int
    
    

class Storyboard(BaseModel):
    id: int
    project_id: Optional[int] = None
    name: Optional[str]
    shots: Optional[List[Shot]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class StoryboardResponse(BaseModel):
    success: bool
    message: str
    storyboards: List[Storyboard]

