from pydantic import BaseModel
from typing import Optional

class ProjectRequest(BaseModel):
    project_name: Optional[str] = "Untitled Project"
    shot_index_size: Optional[int] = 0
    
    
