from pydantic import BaseModel

class ResponseJson(BaseModel):
    success: bool
    message: str
    data: list[dict]
