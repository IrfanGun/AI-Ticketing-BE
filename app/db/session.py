from collections.abc import Generator

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.settings import get_settings


settings = get_settings()
if not settings.database_url:
    raise RuntimeError("DATABASE_URL is not configured.")

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_database() -> None:
    try:
        from app.models.rag import RagChunk, RagDocument  # noqa: F401

        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database is not ready: {exc}") from exc

