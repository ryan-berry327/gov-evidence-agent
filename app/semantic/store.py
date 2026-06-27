"""
In-memory vector store. All chunk embeddings live in one NumPy matrix and search
is a single matrix-vector multiply followed by a top-k selection.

Embeddings are unit-length (see embedder), so the dot product equals cosine
similarity. `matrix @ query` therefore gives the query's similarity to every
chunk at once. Fine for a few thousand chunks; past ~100k you'd move to an ANN
index (FAISS/HNSW) or a managed vector DB behind the same interface.
"""
from __future__ import annotations

import numpy as np

from app.semantic.chunking import Chunk
from app.semantic.embedder import Embedder


class VectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed(texts)
        for c, v in zip(chunks, vectors):
            c.embedding = v
        self.chunks.extend(chunks)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        if not self.chunks:
            self._matrix = None
            return
        self._matrix = np.array([c.embedding for c in self.chunks], dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks most similar to the query, with scores."""
        if self._matrix is None:
            return []
        q = np.array(self.embedder.embed([query])[0], dtype=np.float32)
        scores = self._matrix @ q  # cosine similarity to every chunk at once
        # argpartition is O(n): grab the top_k indices without fully sorting
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]  # sort just those k by score
        return [(self.chunks[i], float(scores[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.chunks)
