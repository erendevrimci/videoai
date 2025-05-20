from pydantic import BaseModel
from typing import Optional

class Script(BaseModel):
    id : int
    title : Optional[str] = None
    topic : Optional[str] = None
    script : Optional[str] = None
    created_at : Optional[str] = None

class ScriptResponse(BaseModel):
    success : bool
    message : str
    script : Optional[Script] = None
    scripts : Optional[list[Script]] = None
