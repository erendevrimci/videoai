from pydantic import BaseModel
from typing import Optional

class CreateProjectResponse(BaseModel):
    success: bool
    message: str
    project_id: Optional[int] = None
    project_name: Optional[str] = None
