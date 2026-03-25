from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, UploadFile
import numpy as np
from sqlalchemy.orm import Session

from app.core.settings import Settings
from app.schemas.rag import DocumentMetadataResponse, DocumentUploadResponse, IndexDocumentResponse, RetrieveResponse
from app.services.chunk_service import ChunkService
from app.services.document_parser import DocumentParserService
from app.services.embedding_service import EmbeddingService
from app.services.rag_repository import RAGRepository


class RAGService:
    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.repository = RAGRepository(db)
        self.document_parser = DocumentParserService()
        self.chunk_service = ChunkService(settings)
        self.embedding_service = EmbeddingService(settings)

    async def upload_document(self, upload: UploadFile) -> DocumentUploadResponse:
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        safe_name = (upload.filename or "document").replace("/", "_").replace("\\", "_")
        stored_path = self.settings.document_dir / f"{uuid4().hex}-{safe_name}"
        stored_path.write_bytes(content)

        document_id = uuid4().hex
        document = self.repository.create_document_record(upload, len(content), stored_path, document_id)
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            status=document.status,
        )

    def get_document(self, document_id: str) -> DocumentMetadataResponse:
        document = self.repository.get_document(document_id)
        return self._to_document_metadata_response(document)

    def list_documents(self) -> list[DocumentMetadataResponse]:
        return [self._to_document_metadata_response(document) for document in self.repository.list_documents()]

    def index_document(self, document_id: str) -> IndexDocumentResponse:
        document = self.repository.get_document(document_id)
        file_path = Path(document.storage_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Stored document file is missing.")

        content = file_path.read_bytes()
        text = self.document_parser.extract_text(document.filename, content)
        chunks = self.chunk_service.split(text, document_id)

        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks were generated from the document.")

        vectors = self.embedding_service.build_vectors(chunks)
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk["embedding"] = vector["values"]

        sample_chunks = [
            {
                "chunk_id": chunk["chunk_id"],
                "text_preview": chunk["text_preview"],
                "token_count": chunk["token_count"],
            }
            for chunk in chunks[:3]
        ]
        document = self.repository.replace_chunks(
            document=document,
            raw_text=text,
            chunks=chunks,
            embedding_model=self.settings.embedding_model,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )

        return IndexDocumentResponse(
            document_id=document.id,
            filename=document.filename,
            raw_text_length=document.raw_text_length,
            chunk_count=document.chunk_count,
            vector_count=document.vector_count,
            chunk_size=document.chunk_size or self.settings.rag_chunk_size,
            chunk_overlap=document.chunk_overlap or self.settings.rag_chunk_overlap,
            embedding_model=document.embedding_model or self.settings.embedding_model,
            index_name="mysql",
            sample_chunks=sample_chunks,
        )

    def retrieve(self, query: str, document_id: str | None, top_k: int) -> RetrieveResponse:
        query_embedding = self.embedding_service.create_embeddings([query])[0]
        chunks = self.repository.load_chunks(document_id)
        if not chunks:
            return RetrieveResponse(query=query, document_id=document_id, total_matches=0, matches=[])

        query_vector = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            raise HTTPException(status_code=422, detail="Query embedding norm is zero.")

        scored_matches: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_vector = np.array(self._parse_embedding(chunk.embedding_json), dtype=np.float32)
            chunk_norm = np.linalg.norm(chunk_vector)
            if chunk_norm == 0:
                continue
            score = float(np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm))
            scored_matches.append(
                {
                    "chunk_id": chunk.id,
                    "score": score,
                    "text_preview": chunk.text_preview,
                    "metadata": {
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "text": chunk.content,
                        "text_preview": chunk.text_preview,
                    },
                }
            )

        normalized_matches = sorted(scored_matches, key=lambda item: item["score"] or 0, reverse=True)[:top_k]

        return RetrieveResponse(
            query=query,
            document_id=document_id,
            total_matches=len(normalized_matches),
            matches=normalized_matches,
        )

    def _to_document_metadata_response(self, document) -> DocumentMetadataResponse:
        sample_chunks = [
            {
                "chunk_id": chunk.id,
                "text_preview": chunk.text_preview,
                "token_count": chunk.token_count,
            }
            for chunk in (document.chunks[:3] if document.chunks else [])
        ]
        return DocumentMetadataResponse(
            document_id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            status=document.status,
            size_bytes=document.size_bytes,
            raw_text_length=document.raw_text_length,
            chunk_count=document.chunk_count,
            vector_count=document.vector_count,
            chunk_size=document.chunk_size,
            chunk_overlap=document.chunk_overlap,
            embedding_model=document.embedding_model,
            index_name="mysql",
            created_at=document.created_at.isoformat() if document.created_at else datetime.now(UTC).isoformat(),
            indexed_at=document.indexed_at.isoformat() if document.indexed_at else None,
            sample_chunks=sample_chunks,
        )

    def _parse_embedding(self, embedding_json: str) -> list[float]:
        import json

        parsed = json.loads(embedding_json)
        if not isinstance(parsed, list):
            raise HTTPException(status_code=500, detail="Stored embedding format is invalid.")
        return [float(value) for value in parsed]
