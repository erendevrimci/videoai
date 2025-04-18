from pydantic import BaseModel
from typing import Optional

class ProjectResponse(BaseModel):
    message: Optional[str] = None
    success: bool
    project_id: Optional[int] = None
    project_name: Optional[str] = None
