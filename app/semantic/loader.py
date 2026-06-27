"""Load a JSON corpus, chunk every document, and return a populated VectorStore."""
from __future__ import annotations

import json
from pathlib import Path

from app.semantic.chunking import chunk_text
from app.semantic.embedder import get_embedder
from app.semantic.store import VectorStore


def load_corpus(path: str | Path) -> VectorStore:
    path = Path(path)
    docs = json.loads(path.read_text())

    store = VectorStore(embedder=get_embedder())
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d["doc_id"], d["title"], d["text"]))
    store.add(all_chunks)
    return store
