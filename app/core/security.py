import time
from collections import defaultdict

from fastapi import Depends, HTTPException, Request

from app.core.settings import Settings, get_settings


_requests_log: dict[str, list[float]] = defaultdict(list)


def rate_limiter(client_id: str, settings: Settings) -> None:
    now = time.time()
    window_start = now - 60
    history = [item for item in _requests_log.get(client_id, []) if item >= window_start]

    if len(history) >= settings.rate_limit_per_min:
        raise HTTPException(status_code=429, detail="Too many requests")

    history.append(now)
    _requests_log[client_id] = history


def require_api_key(request: Request, settings: Settings = Depends(get_settings)) -> None:
    if settings.api_key is None:
        return

    client_key = request.headers.get("x-api-key")

    if client_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    client_id = request.client.host if request.client else "unknown"
    rate_limiter(client_id, settings)

