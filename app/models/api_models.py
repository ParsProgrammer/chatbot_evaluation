from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    latency_ms: Optional[float] = None