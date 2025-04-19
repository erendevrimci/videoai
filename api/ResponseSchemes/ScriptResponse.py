from pydantic import BaseModel
from typing import Optional

class Script(BaseModel):
    id : int
    title : str
    topic : str
    script : str

class ScriptResponse(BaseModel):
    success : bool
    message : str
    script : Optional[Script] = None
    scripts : Optional[list[Script]] = None
