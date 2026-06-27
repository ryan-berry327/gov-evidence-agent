# Government Evidence Agent

A citation-backed question-answering service over a document corpus. A question
goes in; the agent runs its own searches over a semantic layer, gathers evidence,
and returns a concise answer where every claim cites the passage it came from.

Provider-agnostic: it runs on a local Ollama model for development and CI, and on
an API model in production, behind one interface with no changes to the agent.

---

## What it does

1. **Semantic layer** — documents are chunked, embedded, and stored as vectors;
   retrieval is cosine similarity over those vectors.
2. **Agent** — an LLM in a bounded loop that decides when to search, can search
   more than once, and returns a cited answer or abstains.
3. **API** — a FastAPI service (`/ask`, `/health`) with typed request/response
   models, structured logging, and an error boundary.
4. **Evaluation** — retrieval metrics (hit@k, recall@k, MRR) and answer metrics
   (fact-match, LLM-as-judge faithfulness, abstention), with a CI gate that fails
   the build on quality regression.
5. **Deploy & monitor** — containerised, deployed to Azure Container Apps via
   Terraform, instrumented with OpenTelemetry into Application Insights.
6. **CI/CD** — GitHub Actions: tests + eval gate on every push; build, push and
   `terraform apply` on main.

---

## Running locally

```bash
pip install -r requirements.txt

# Tests + eval gate — no model, no network:
pytest tests/ -v
python -m app.eval.run            # retrieval metrics

# Full agent on a local model:
#   install Ollama (https://ollama.com), then:
ollama pull llama3.1
python demo.py "How much annual leave do civil servants get?"

# Run the API:
uvicorn app.api.main:app --reload
#   then open http://localhost:8000/docs
```

Switch to the production model with `LLM_PROVIDER=anthropic` and an API key.

---

## Architecture

```
question
   │
   ▼
FastAPI (/ask)  ──logs/metrics──►  App Insights (OpenTelemetry)
   │
   ▼
EvidenceAgent  ── bounded loop, max N steps ──┐
   │  picks queries, decides when to answer   │
   ▼                                          │
search_evidence tool                          │
   │                                          │
   ▼                                          │
VectorStore (cosine similarity, NumPy)        │
   │                                          │
   ▼                                          ▼
top-k chunks ──────────────────────►  cited answer + audit trail
```

---

## Layout

| Area | Where |
|---|---|
| Semantic layer (chunking, embedding, search) | `app/semantic/` |
| Agent loop and LLM providers | `app/agent/` |
| API service | `app/api/` |
| Evaluation and CI gate | `app/eval/`, `tests/test_eval_gate.py` |
| Container image | `Dockerfile` |
| Infrastructure | `infra/main.tf` |
| CI/CD | `.github/workflows/` |

---

## Limitations and next steps

- The local dev embedder is a word-overlap baseline, not a semantic model — it's
  there for fast, deterministic CI. Production uses a real embedding model
  (`USE_REAL_EMBED=1`); the eval harness quantifies the difference.
- Vector search is an in-memory linear scan. Past ~100k chunks, swap in an ANN
  index (FAISS/HNSW) or a managed vector DB behind the same interface.
- The LLM-as-judge should be validated against human labels on a sample before
  its scores are trusted.
- The deploy workflow and Terraform are defined but a full end-to-end deploy
  needs a real Azure subscription. Next: a manual prod approval gate and a
  post-deploy smoke test before shifting traffic.
- Single-tool agent (corpus search only). A natural extension is a second
  web-search tool with an eval showing the agent routes correctly between
  internal evidence and the open web.
