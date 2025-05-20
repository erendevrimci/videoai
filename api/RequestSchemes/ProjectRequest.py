from pydantic import BaseModel
from typing import Optional

class ProjectRequest(BaseModel):
    project_name: Optional[str] = "Untitled Project"
    
    
