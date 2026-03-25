from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    app_name: str = "AI Financial Service"
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://127.0.0.1:8000"
    api_key: str | None = Field(default=None, alias="API_KEY")
    rate_limit_per_min: int = Field(default=60, alias="RATE_LIMIT_PER_MIN")

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    ai_model: str = Field(default="llama-3.3-70b-versatile", alias="AI_MODEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    rag_chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP")
    rag_retrieval_top_k: int = Field(default=5, alias="RAG_RETRIEVAL_TOP_K")

    storage_dir: Path = Field(default=ROOT_DIR / "storage", alias="STORAGE_DIR")

    @property
    def document_dir(self) -> Path:
        return self.storage_dir / "documents"

    @property
    def metadata_dir(self) -> Path:
        return self.storage_dir / "metadata"

    @property
    def chunk_dir(self) -> Path:
        return self.storage_dir / "chunks"

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.document_dir.mkdir(parents=True, exist_ok=True)
    return settings
