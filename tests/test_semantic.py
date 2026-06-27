"""Tests for the semantic layer: chunking produces overlapping windows, and
search returns results in descending similarity order."""
from app.semantic.chunking import chunk_text
from app.semantic.loader import load_corpus


def test_chunking_overlaps_and_covers():
    text = "word " * 500  # ~2500 chars, forces multiple chunks
    chunks = chunk_text("d1", "Doc One", text)
    assert len(chunks) > 1
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.text for c in chunks)  # no empty chunks
    # chunk ids are sequential from 0
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_search_returns_sorted_scores():
    store = load_corpus("corpus/sample_corpus.json")
    results = store.search("pension contributions", top_k=3)
    assert len(results) <= 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)  # descending


def test_citation_format():
    chunks = chunk_text("d1", "My Doc", "hello world")
    assert chunks[0].citation == "[My Doc #0]"
