from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: str


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    status: str
    size_bytes: int
    raw_text_length: int
    chunk_count: int
    vector_count: int
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_model: str | None = None
    index_name: str | None = None
    created_at: str
    indexed_at: str | None = None
    sample_chunks: list[dict[str, Any]] = Field(default_factory=list)


class IndexDocumentResponse(BaseModel):
    document_id: str
    filename: str
    raw_text_length: int
    chunk_count: int
    vector_count: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    index_name: str
    sample_chunks: list[dict[str, Any]]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveMatch(BaseModel):
    chunk_id: str
    score: float | None = None
    text_preview: str
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    document_id: str | None = None
    total_matches: int
    matches: list[RetrieveMatch]

