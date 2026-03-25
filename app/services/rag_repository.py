import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session, selectinload

from app.models.rag import RagChunk, RagDocument


class RAGRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_document_record(self, upload: UploadFile, size_bytes: int, stored_path: Path, document_id: str) -> RagDocument:
        document = RagDocument(
            id=document_id,
            filename=upload.filename or "document",
            content_type=upload.content_type or "application/octet-stream",
            storage_path=str(stored_path),
            status="uploaded",
            size_bytes=size_bytes,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_document(self, document_id: str) -> RagDocument:
        document = (
            self.db.query(RagDocument)
            .options(selectinload(RagDocument.chunks))
            .filter(RagDocument.id == document_id)
            .first()
        )
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        return document

    def list_documents(self) -> list[RagDocument]:
        return self.db.query(RagDocument).order_by(RagDocument.created_at.desc()).all()

    def replace_chunks(
        self,
        document: RagDocument,
        raw_text: str,
        chunks: list[dict[str, Any]],
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> RagDocument:
        self.db.query(RagChunk).filter(RagChunk.document_id == document.id).delete()

        for chunk in chunks:
            self.db.add(
                RagChunk(
                    id=str(chunk["chunk_id"]),
                    document_id=document.id,
                    chunk_index=int(chunk["chunk_index"]),
                    token_count=int(chunk["token_count"]),
                    content=str(chunk["text"]),
                    text_preview=str(chunk["text_preview"]),
                    embedding_json=json.dumps(chunk["embedding"]),
                )
            )

        document.status = "indexed"
        document.raw_text = raw_text
        document.raw_text_length = len(raw_text)
        document.chunk_count = len(chunks)
        document.vector_count = len(chunks)
        document.chunk_size = chunk_size
        document.chunk_overlap = chunk_overlap
        document.embedding_model = embedding_model
        document.indexed_at = datetime.now(UTC)

        self.db.commit()
        self.db.refresh(document)
        return document

    def load_chunks(self, document_id: str | None = None) -> list[RagChunk]:
        query = self.db.query(RagChunk)
        if document_id:
            query = query.filter(RagChunk.document_id == document_id)
        return query.order_by(RagChunk.document_id.asc(), RagChunk.chunk_index.asc()).all()
