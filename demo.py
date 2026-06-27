"""
Local demo.

Setup (once): install Ollama from https://ollama.com, then `ollama pull llama3.1`
and make sure `ollama serve` is running.

    python demo.py "How much annual leave do civil servants get?"

Uses the local Ollama model and local embedder by default. For the production
path:
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python demo.py "..."
"""
import sys

from app.semantic.loader import load_corpus
from app.agent.agent import EvidenceAgent


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "How much annual leave do civil servants get?"

    print(f"\nQuestion: {question}\n")
    store = load_corpus("corpus/sample_corpus.json")
    print(f"Corpus loaded: {len(store)} chunks\n")

    agent = EvidenceAgent(store=store)  # provider chosen from env (Ollama default)
    result = agent.answer(question)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    print()
    print(f"(took {result.steps} agent steps)")
    print()
    print("=" * 70)
    print("EVIDENCE THE AGENT RETRIEVED (audit trail)")
    print("=" * 70)
    for e in result.evidence:
        print(f"  {e['citation']}  score={e['score']}")
    print()
    print("SEARCH TRACE:")
    for t in result.trace:
        print(f"  step {t['step']}: searched {t['query']!r} -> {t['n_hits']} hits")


if __name__ == "__main__":
    main()
