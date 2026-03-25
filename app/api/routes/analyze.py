from fastapi import APIRouter, Depends

from app.core.security import require_api_key
from app.core.settings import Settings, get_settings
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.ai_chat_service import AIChatService


router = APIRouter(prefix="", tags=["AI"])


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(require_api_key)])
def analyze(request: AnalyzeRequest, settings: Settings = Depends(get_settings)) -> AnalyzeResponse:
    return AIChatService(settings).analyze(request)

