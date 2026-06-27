"""
Request/response models. FastAPI uses these to validate input (a malformed body
returns a 422 instead of crashing) and to generate the OpenAPI docs at /docs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The policy question to answer from the corpus.")


class EvidenceItem(BaseModel):
    citation: str
    doc_title: str
    chunk_id: int
    text: str
    score: float


class TraceItem(BaseModel):
    step: int
    query: str
    n_hits: int


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    steps: int
    trace: list[TraceItem]


class HealthResponse(BaseModel):
    status: str
    corpus_chunks: int
    llm_provider: str
