from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.core.settings import Settings, get_settings
from app.db.session import get_db
from app.schemas.rag import (
    DocumentMetadataResponse,
    DocumentUploadResponse,
    IndexDocumentResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.rag_service import RAGService


router = APIRouter(prefix="/rag", tags=["RAG"], dependencies=[Depends(require_api_key)])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    return await RAGService(settings, db).upload_document(file)


@router.get("/documents", response_model=list[DocumentMetadataResponse])
def list_documents(settings: Settings = Depends(get_settings), db: Session = Depends(get_db)) -> list[DocumentMetadataResponse]:
    return RAGService(settings, db).list_documents()


@router.get("/documents/{document_id}", response_model=DocumentMetadataResponse)
def get_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> DocumentMetadataResponse:
    return RAGService(settings, db).get_document(document_id)


@router.post("/documents/{document_id}/index", response_model=IndexDocumentResponse)
def index_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> IndexDocumentResponse:
    return RAGService(settings, db).index_document(document_id)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    request: RetrieveRequest,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> RetrieveResponse:
    return RAGService(settings, db).retrieve(
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k or settings.rag_retrieval_top_k,
    )
