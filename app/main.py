from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analyze import router as analyze_router
from app.api.routes.rag import router as rag_router
from app.core.settings import get_settings
from app.db.session import ensure_database


settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(rag_router)


@app.on_event("startup")
def startup() -> None:
    ensure_database()


@app.get("/healthz", tags=["Health"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["Health"])
def readyz() -> dict[str, str]:
    return {"status": "ready"}
