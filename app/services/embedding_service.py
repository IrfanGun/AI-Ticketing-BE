from typing import Any

from fastapi import HTTPException
from openai import OpenAI

from app.core.settings import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.client is None:
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not configured. Embedding service is unavailable.",
            )

        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def build_vectors(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts = [str(chunk["text"]) for chunk in chunks]
        embeddings = self.create_embeddings(texts)

        vectors: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vectors.append(
                {
                    "id": str(chunk["chunk_id"]),
                    "values": embedding,
                    "metadata": {
                        "document_id": str(chunk["document_id"]),
                        "chunk_index": int(chunk["chunk_index"]),
                        "token_count": int(chunk["token_count"]),
                        "text": str(chunk["text"]),
                        "text_preview": str(chunk["text_preview"]),
                    },
                }
            )
        return vectors

