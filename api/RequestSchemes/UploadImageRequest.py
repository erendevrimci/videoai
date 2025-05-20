from pydantic import BaseModel


class UploadImageRequest(BaseModel):
    image: str
    name: str
