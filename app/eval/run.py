"""
Eval runner. Two modes:

  retrieval-only  : semantic layer only. No LLM, fast and deterministic, so it
                    runs in CI as a quality gate.
  full            : runs the whole agent per question (needs a provider, e.g.
                    local Ollama) and adds answer-quality + faithfulness metrics.

Run:
    python -m app.eval.run                # retrieval-only (default)
    python -m app.eval.run --full         # full agent eval (uses configured LLM)

The retrieval-only mode can fail the build if hit@k drops below a threshold,
catching a regression in the semantic layer with no model and no cost.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from app.config import settings
from app.semantic.loader import load_corpus
from app.eval import retrieval_metrics as rm
from app.eval import answer_metrics as am

DATASET = Path(__file__).parent / "dataset.json"
CORPUS = Path(__file__).parents[2] / "corpus" / "sample_corpus.json"


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaNs (unanswerable questions)
    return round(statistics.mean(xs), 3) if xs else float("nan")


def run_retrieval_only(k: int | None = None) -> dict:
    k = k or settings.top_k
    store = load_corpus(CORPUS)
    cases = json.loads(DATASET.read_text())

    hits, recalls, rrs = [], [], []
    for c in cases:
        if not c["answerable"]:
            continue
        retrieved = rm.retrieved_doc_ids(store, c["question"], k)
        rel = c["relevant_doc_ids"]
        hits.append(rm.hit_at_k(retrieved, rel))
        recalls.append(rm.recall_at_k(retrieved, rel))
        rrs.append(rm.reciprocal_rank(retrieved, rel))

    return {
        "mode": "retrieval_only",
        "k": k,
        "n": len(hits),
        "hit@k": _mean(hits),
        "recall@k": _mean(recalls),
        "mrr": _mean(rrs),
    }


def run_full() -> dict:
    from app.agent.agent import EvidenceAgent
    from app.agent.llm import get_llm

    store = load_corpus(CORPUS)
    cases = json.loads(DATASET.read_text())
    judge = get_llm()
    agent = EvidenceAgent(store=store)

    fact_scores, faith_scores, cite_flags = [], [], []
    abstain_correct = []

    for c in cases:
        result = agent.answer(c["question"])
        if c["answerable"]:
            fact_scores.append(am.fact_match(result.answer, c["expected_facts"]))
            cite_flags.append(1.0 if am.cites_evidence(result.answer) else 0.0)
            ev_texts = [e["text"] for e in result.evidence]
            f = am.llm_faithfulness(result.answer, ev_texts, judge)
            if f is not None:
                faith_scores.append(f)
        else:
            abstain_correct.append(1.0 if am.is_abstention(result.answer) else 0.0)

    return {
        "mode": "full",
        "fact_match": _mean(fact_scores),
        "citation_rate": _mean(cite_flags),
        "faithfulness_llm_judge": _mean(faith_scores),
        "abstention_accuracy": _mean(abstain_correct),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run full agent eval (needs LLM)")
    args = ap.parse_args()
    report = run_full() if args.full else run_retrieval_only()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
