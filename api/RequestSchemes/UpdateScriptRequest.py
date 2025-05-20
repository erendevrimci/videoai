from pydantic import BaseModel

class UpdateScriptRequest(BaseModel):
    script_id: int
    script: str


