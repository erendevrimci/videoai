from pydantic import BaseModel
from typing import Optional

class ScriptRequest(BaseModel):
    # user_id artık opsiyonel, çünkü JWT token'dan otomatik olarak alınacak
    user_id: Optional[str] = None
    channel_number: Optional[int] = None
    topic: str
    context: str
