"""
Retrieval metrics for the semantic layer: when we search, do the right documents
come back? No LLM involved, so these are cheap and deterministic.

- hit@k:    did at least one relevant doc appear in the top k? (binary per query)
- recall@k: of the relevant docs for a query, what fraction did we retrieve in top k?
- MRR:      1/rank of the first relevant doc, averaged.

Evaluated at the document level, matching how the labels are written
(relevant_doc_ids).
"""
from __future__ import annotations

from app.semantic.store import VectorStore


def retrieved_doc_ids(store: VectorStore, query: str, k: int) -> list[str]:
    """Doc ids of the top-k retrieved chunks, in rank order, de-duplicated."""
    hits = store.search(query, top_k=k)
    seen, ordered = set(), []
    for chunk, _score in hits:
        if chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            ordered.append(chunk.doc_id)
    return ordered


def hit_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return float("nan")  # undefined for unanswerable questions
    return 1.0 if any(r in relevant for r in retrieved) else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return float("nan")
    found = sum(1 for r in relevant if r in retrieved)
    return found / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return float("nan")
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0
