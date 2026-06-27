"""
FastAPI service wrapping the evidence agent.

Notes:
- The corpus is chunked and embedded once at startup (lifespan handler), not per
  request. The VectorStore lives in app.state and is reused.
- /health reports liveness and whether the corpus is loaded, for orchestrator
  health probes.
- Each request logs latency and step count as structured fields for monitoring.
- The agent runs inside try/except so a provider failure returns a 503 rather
  than leaking a stack trace.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.config import settings
from app.semantic.loader import load_corpus
from app.agent.agent import EvidenceAgent
from app.api.schemas import AskRequest, AskResponse, HealthResponse
from app.api.monitoring import setup_monitoring, record_request_metrics

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
)
logger = logging.getLogger("gov-evidence-agent")

CORPUS = Path(__file__).parents[2] / "corpus" / "sample_corpus.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: build the semantic layer once.
    logger.info("loading corpus and building semantic layer")
    store = load_corpus(CORPUS)
    app.state.store = store
    app.state.agent = EvidenceAgent(store=store)
    logger.info(f"ready: {len(store)} chunks, provider={settings.llm_provider}")
    yield
    # Shutdown: nothing to clean up for the in-memory store.
    logger.info("shutting down")


app = FastAPI(
    title="Government Evidence Agent",
    description="Agentic, cited question-answering over a document corpus.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable cloud telemetry if configured (no-op without a connection string).
setup_monitoring(app)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    store = request.app.state.store
    return HealthResponse(
        status="ok",
        corpus_chunks=len(store),
        llm_provider=settings.llm_provider,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    agent: EvidenceAgent = request.app.state.agent
    start = time.perf_counter()
    try:
        result = agent.answer(body.question)
    except Exception as e:
        logger.exception("agent failure")
        raise HTTPException(status_code=503, detail="agent backend unavailable") from e
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info(
        f'request handled latency_ms={elapsed_ms} steps={result.steps} '
        f'evidence={len(result.evidence)} q_len={len(body.question)}'
    )
    record_request_metrics(elapsed_ms, result.steps, len(result.evidence))

    return AskResponse(
        answer=result.answer,
        evidence=result.evidence,
        steps=result.steps,
        trace=result.trace,
    )
