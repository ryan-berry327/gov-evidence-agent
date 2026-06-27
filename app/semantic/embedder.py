"""
Text -> vector, behind a small Embedder protocol so the rest of the app doesn't
care which backend is used.

- HashingEmbedder: local, dependency-free, deterministic. Hashes words into fixed
  dimensions and counts them, so texts sharing words get similar vectors. It is a
  word-overlap baseline, not a real semantic model. Used by default for dev/tests
  so everything runs with no API key and no cost.
- VoyageEmbedder: real embedding API. Enable with USE_REAL_EMBED=1.
"""
from __future__ import annotations
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic local embedder. No network, no cost."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in _WORD.findall(t.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm  # unit length, so dot product == cosine similarity
            out.append(vec.tolist())
        return out


class VoyageEmbedder:
    """Real embedding API. Requires VOYAGE_API_KEY."""

    def __init__(self, model: str = "voyage-3"):
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        import os

        key = os.environ["VOYAGE_API_KEY"]
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"input": texts, "model": self.model},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in data]


def get_embedder() -> Embedder:
    import os

    if os.environ.get("USE_REAL_EMBED") == "1":
        return VoyageEmbedder()
    return HashingEmbedder()
