from pydantic import BaseModel
from typing import Optional

class CreateStoryboardResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None