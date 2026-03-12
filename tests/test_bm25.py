"""
Unit tests for BM25 index: build, query, ordering, persistence.
"""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Toy corpus: 3 docs with deterministic expected ordering
TOY_CORPUS = [
    {"doc_id": "doc_001", "title": "Python programming language", "text": "Python is a high-level general-purpose programming language known for its readability and simplicity."},
    {"doc_id": "doc_002", "title": "Machine learning basics", "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
    {"doc_id": "doc_003", "title": "Database management systems", "text": "A database management system is software that interacts with end users applications and the database itself."},
]


@pytest.fixture
def bm25():
    """Fresh BM25Index instance (not the module singleton)."""
    from app.search.bm25_search import BM25Index
    idx = BM25Index()
    idx.build(TOY_CORPUS)
    return idx


class TestBM25Build:
    def test_build_sets_loaded(self, bm25):
        assert bm25.is_loaded is True

    def test_num_docs(self, bm25):
        assert bm25.num_docs == 3


class TestBM25Query:
    def test_python_query_returns_python_doc_first(self, bm25):
        results = bm25.query("Python programming", top_k=3)
        top_doc_id = results[0][0]
        assert top_doc_id == "doc_001", f"Expected doc_001 first, got {top_doc_id}"

    def test_machine_learning_query(self, bm25):
        results = bm25.query("machine learning artificial intelligence", top_k=3)
        doc_ids = [r[0] for r in results]
        assert "doc_002" in doc_ids

    def test_database_query(self, bm25):
        results = bm25.query("database management software", top_k=3)
        assert results[0][0] == "doc_003"

    def test_scores_non_negative(self, bm25):
        results = bm25.query("programming", top_k=3)
        for _, score in results:
            assert score >= 0.0

    def test_empty_query_returns_empty(self, bm25):
        # Tokenizer removes noise; very short/empty queries return no results
        results = bm25.query("", top_k=3)
        assert results == []

    def test_top_k_respected(self, bm25):
        results = bm25.query("data", top_k=2)
        assert len(results) <= 2

    def test_ordering_is_descending(self, bm25):
        results = bm25.query("programming language", top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25Persistence:
    def test_save_and_load(self, bm25, tmp_path):
        from app.search.bm25_search import BM25_PICKLE, BM25_META, BM25_DOCIDS
        import app.search.bm25_search as bm25_mod

        # Patch paths to tmp_path
        with patch.object(bm25_mod, "BM25_PICKLE", tmp_path / "bm25.pkl"), \
             patch.object(bm25_mod, "BM25_DOCIDS", tmp_path / "doc_ids.json"), \
             patch.object(bm25_mod, "BM25_META", tmp_path / "meta.json"):
            bm25.save()

            from app.search.bm25_search import BM25Index
            idx2 = BM25Index()
            with patch.object(bm25_mod, "BM25_PICKLE", tmp_path / "bm25.pkl"), \
                 patch.object(bm25_mod, "BM25_DOCIDS", tmp_path / "doc_ids.json"), \
                 patch.object(bm25_mod, "BM25_META", tmp_path / "meta.json"):
                idx2.load()

            assert idx2.num_docs == 3
            results = idx2.query("Python programming", top_k=3)
            assert results[0][0] == "doc_001"
