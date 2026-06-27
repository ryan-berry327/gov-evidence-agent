"""
Document model and chunking.

A chunk is a small, retrievable slice of a document, so we can return the
paragraph that answers a question with a citation rather than a whole report.

Strategy: fixed-size character windows with overlap, so a sentence straddling a
boundary still appears whole in one chunk. Splitting on headings or sentences
would be smarter; fixed windows are a simple, predictable baseline.
"""
from dataclasses import dataclass, field
from app.config import settings


@dataclass
class Chunk:
    doc_id: str          # which source document
    doc_title: str       # human-readable, used in citations
    chunk_id: int        # position within the document
    text: str            # the actual content
    embedding: list[float] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"[{self.doc_title} #{self.chunk_id}]"


def chunk_text(doc_id: str, doc_title: str, text: str) -> list[Chunk]:
    """Slice a document into overlapping character windows."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    step = size - overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    cid = 0
    text = text.strip()
    while start < len(text):
        window = text[start : start + size]
        if window.strip():
            chunks.append(
                Chunk(doc_id=doc_id, doc_title=doc_title, chunk_id=cid, text=window.strip())
            )
            cid += 1
        start += step
    return chunks
