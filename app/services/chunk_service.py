import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.settings import Settings


class ChunkService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            self.encoding = tiktoken.encoding_for_model(settings.embedding_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def split(self, text: str, document_id: str) -> list[dict[str, object]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
            length_function=self.token_count,
        )

        chunks = splitter.split_text(text)
        return [
            {
                "chunk_id": f"{document_id}-chunk-{index + 1}",
                "document_id": document_id,
                "text": chunk,
                "text_preview": chunk[:240].strip(),
                "token_count": self.token_count(chunk),
                "chunk_index": index,
            }
            for index, chunk in enumerate(chunks)
        ]

    def token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

