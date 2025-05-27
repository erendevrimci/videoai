from pydantic import BaseModel
from typing import Optional

class ScriptRequest(BaseModel):
    project_id: int
    topic: str
    context: str
    channel_number: Optional[int] = 1
    script: Optional[str] = None
    script_id: Optional[int] = None
    tone: Optional[str] = "informative"
