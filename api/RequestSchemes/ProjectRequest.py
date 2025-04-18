from pydantic import BaseModel
from typing import Optional

class ProjectRequest(BaseModel):
    project_name: Optional[str] = "Untitled Project"
    script_id: Optional[int] = None
    voice_over_id: Optional[int] = None
    caption_id: Optional[int] = None
    final_video_id: Optional[int] = None
    
