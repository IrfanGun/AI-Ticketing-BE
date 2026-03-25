from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    type: str = Field(min_length=1, max_length=64)
    data: dict[str, Any]
    options: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    answer: str
    provider: str = "groq"
    model: str
    context_summary: str | None = None

