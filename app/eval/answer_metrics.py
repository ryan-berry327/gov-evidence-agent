"""
Answer-quality metrics, measured on the final answer given the retrieved evidence.

- fact_match: does the answer contain the expected key facts (e.g. "25 days")?
  Deterministic and free; brittle to wording, so it's a regression guardrail, not
  a judgement of quality.
- is_abstention: for unanswerable questions, did the answer correctly refuse?
- cites_evidence: does the answer carry at least one [Title #n] citation?
- llm_faithfulness: an LLM-as-judge score for whether every claim is supported by
  the evidence. The judge is itself a model, so validate it against human labels
  on a sample before trusting its scores.
"""
from __future__ import annotations

import re

_NOT_FOUND_SIGNALS = [
    "don't have", "do not have", "not contain", "no information", "cannot find",
    "can't find", "not found", "unable to", "does not appear", "no relevant",
    "not available", "not in the", "insufficient",
]


def fact_match(answer: str, expected_facts: list[str]) -> float:
    """Fraction of expected facts present in the answer (case-insensitive)."""
    if not expected_facts:
        return float("nan")
    a = answer.lower()
    hits = sum(1 for f in expected_facts if f.lower() in a)
    return hits / len(expected_facts)


def is_abstention(answer: str) -> bool:
    """True if the answer declines to answer (correct for unanswerable Qs)."""
    a = answer.lower()
    return any(sig in a for sig in _NOT_FOUND_SIGNALS)


def cites_evidence(answer: str) -> bool:
    """True if the answer contains at least one [Title #n] citation marker."""
    return bool(re.search(r"\[[^\]]+#\d+\]", answer))


# ---- LLM-as-judge (optional; uses whichever provider is configured) --------
JUDGE_PROMPT = """You are grading whether an ANSWER is fully supported by the EVIDENCE.

Respond with ONLY a single integer 1-5 and nothing else:
5 = every claim in the answer is directly supported by the evidence
3 = partially supported; some claims unsupported or vague
1 = answer contradicts or is unsupported by the evidence

EVIDENCE:
{evidence}

ANSWER:
{answer}
"""


def llm_faithfulness(answer: str, evidence_texts: list[str], judge) -> float | None:
    """Score 0-1 via an LLM judge. `judge` is any LLMClient. Returns None on error."""
    if not evidence_texts:
        return None
    prompt = JUDGE_PROMPT.format(
        evidence="\n\n".join(evidence_texts), answer=answer
    )
    try:
        resp = judge.complete(system="You are a strict grader.",
                              messages=[{"role": "user", "content": prompt}], tools=[])
        m = re.search(r"[1-5]", resp.text)
        if not m:
            return None
        return (int(m.group()) - 1) / 4.0  # map 1..5 -> 0..1
    except Exception:
        return None
