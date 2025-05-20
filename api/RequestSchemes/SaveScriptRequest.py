from pydantic import BaseModel

class SaveScriptRequest(BaseModel):
    project_id: int
    script: str
    

