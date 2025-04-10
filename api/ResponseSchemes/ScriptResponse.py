from pydantic import BaseModel
from typing import Optional

class Script(BaseModel):
    id : int
    user_id : str
    script : str
    title : str
    topic : str
    channel_number : int

class ScriptResponse(BaseModel):
    success : bool
    message : str
    script : Optional[Script] = None
    scripts : Optional[list[Script]] = None
