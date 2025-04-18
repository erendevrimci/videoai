from pydantic import BaseModel
from typing import Optional



class ScriptResponse(BaseModel):
    success : bool
    message : str
    script : Optional[str] = None
    scripts : Optional[list[str]] = None
