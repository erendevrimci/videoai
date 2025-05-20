from pydantic import BaseModel
from typing import Optional

class Project(BaseModel):
    id: int
    name: str
    
class ProjectResponse(BaseModel):
    message: Optional[str] = None
    success: bool
    projects: Optional[list[Project]] = None
