from io import BytesIO

import pdfplumber
from fastapi import HTTPException
from pypdf import PdfReader


class DocumentParserService:
    def extract_text(self, filename: str, content: bytes) -> str:
        lowered = filename.lower()

        if lowered.endswith(".pdf"):
            return self._extract_pdf_text(content)

        if lowered.endswith(".txt") or lowered.endswith(".md") or lowered.endswith(".csv"):
            return content.decode("utf-8", errors="ignore").strip()

        raise HTTPException(status_code=415, detail="Unsupported document type. Use PDF, TXT, MD, or CSV.")

    def _extract_pdf_text(self, content: bytes) -> str:
        text_parts: list[str] = []

        reader = PdfReader(BytesIO(content))
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")

        merged = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
        if merged:
            return merged

        with pdfplumber.open(BytesIO(content)) as pdf:
            plumber_parts = [(page.extract_text() or "").strip() for page in pdf.pages]

        merged = "\n".join(part for part in plumber_parts if part).strip()
        if merged:
            return merged

        raise HTTPException(status_code=422, detail="Unable to extract text from the uploaded PDF.")

