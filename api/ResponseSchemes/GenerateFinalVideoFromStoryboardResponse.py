from pydantic import BaseModel
from typing import Optional

class GenerateFinalVideoFromStoryboardResponse(BaseModel):
    success: bool
    message: str
    video_url: Optional[str] = None 
    task_id: Optional[str] = None 