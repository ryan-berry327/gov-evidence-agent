"""
Quality gate: the retrieval eval must clear a threshold, or the build FAILS.
This is how an eval harness becomes a guardrail rather than a one-off script —
a regression in chunking, embedding, or search that drops retrieval quality
will now break CI. No LLM or network required, so it runs free in the pipeline.
"""
from app.eval.run import run_retrieval_only


def test_retrieval_quality_gate():
    report = run_retrieval_only(k=3)
    # If a change tanks retrieval, these assertions trip and the build goes red.
    assert report["hit@k"] >= 0.85, f"hit@k regressed: {report}"
    assert report["mrr"] >= 0.80, f"MRR regressed: {report}"
