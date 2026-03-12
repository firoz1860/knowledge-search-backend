"""
Tests for hybrid scoring, normalization strategies, and Scenario C (divide-by-zero).
"""
import pytest
import math
from unittest.mock import patch, MagicMock

from app.search.hybrid import (
    _minmax_normalize,
    _rank_normalize,
    hybrid_search,
    set_doc_store,
)


class TestMinMaxNormalize:
    def test_normal_range(self):
        scores = {"a": 0.0, "b": 0.5, "c": 1.0}
        normed = _minmax_normalize(scores)
        assert normed["a"] == pytest.approx(0.0)
        assert normed["b"] == pytest.approx(0.5)
        assert normed["c"] == pytest.approx(1.0)

    def test_all_equal_no_nan(self):
        """Scenario C: all scores identical — must NOT produce NaN."""
        scores = {"a": 5.0, "b": 5.0, "c": 5.0}
        normed = _minmax_normalize(scores)
        for v in normed.values():
            assert not math.isnan(v), "NaN detected in normalized scores!"
            assert v == 0.0

    def test_all_zero_no_nan(self):
        """Edge case: all-zero scores."""
        scores = {"a": 0.0, "b": 0.0}
        normed = _minmax_normalize(scores)
        for v in normed.values():
            assert not math.isnan(v)

    def test_empty_input(self):
        assert _minmax_normalize({}) == {}

    def test_single_entry(self):
        scores = {"only": 3.7}
        normed = _minmax_normalize(scores)
        assert not math.isnan(normed["only"])


class TestRankNormalize:
    def test_descending_scores(self):
        scored_list = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
        normed = _rank_normalize(scored_list)
        # Rank 0 → 1/(60+1), rank 1 → 1/(60+2), etc.
        assert normed["a"] > normed["b"] > normed["c"]

    def test_all_positive(self):
        scored_list = [("x", 0.0), ("y", 0.0)]  # same raw score
        normed = _rank_normalize(scored_list)
        for v in normed.values():
            assert v > 0.0
            assert not math.isnan(v)


class TestHybridSearch:
    def _make_docs(self):
        return [
            {"doc_id": "doc_001", "title": "Python guide", "text": "Python programming language tutorial."},
            {"doc_id": "doc_002", "title": "Machine Learning", "text": "Supervised and unsupervised learning methods."},
            {"doc_id": "doc_003", "title": "Databases", "text": "SQL and NoSQL database systems overview."},
        ]

    def test_hybrid_score_in_0_1(self):
        docs = self._make_docs()

        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = [("doc_001", 2.5), ("doc_002", 1.0), ("doc_003", 0.5)]

        mock_vec = MagicMock()
        mock_vec.query.return_value = [("doc_002", 0.9), ("doc_001", 0.7), ("doc_003", 0.4)]

        set_doc_store(docs)

        with patch("app.search.hybrid.bm25_index", mock_bm25), \
             patch("app.search.hybrid.vector_index", mock_vec):
            results = hybrid_search("python machine learning", top_k=3, alpha=0.5)

        for r in results:
            assert 0.0 <= r["hybrid_score"] <= 1.0 + 1e-6, f"Out of range: {r['hybrid_score']}"
            assert not math.isnan(r["hybrid_score"])

    def test_alpha_1_pure_bm25(self):
        """alpha=1.0 should rank purely by BM25."""
        docs = self._make_docs()
        set_doc_store(docs)

        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = [("doc_003", 5.0), ("doc_001", 2.0), ("doc_002", 1.0)]
        mock_vec = MagicMock()
        mock_vec.query.return_value = [("doc_001", 0.95), ("doc_002", 0.8), ("doc_003", 0.2)]

        with patch("app.search.hybrid.bm25_index", mock_bm25), \
             patch("app.search.hybrid.vector_index", mock_vec):
            results = hybrid_search("query", top_k=3, alpha=1.0)

        assert results[0]["doc_id"] == "doc_003"

    def test_no_nan_when_all_scores_equal(self):
        """Regression for Scenario C: same score triggers divide-by-zero."""
        docs = self._make_docs()
        set_doc_store(docs)

        # All BM25 scores identical → old code would produce NaN
        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = [("doc_001", 3.0), ("doc_002", 3.0), ("doc_003", 3.0)]
        mock_vec = MagicMock()
        mock_vec.query.return_value = [("doc_001", 0.8), ("doc_002", 0.6), ("doc_003", 0.4)]

        with patch("app.search.hybrid.bm25_index", mock_bm25), \
             patch("app.search.hybrid.vector_index", mock_vec):
            results = hybrid_search("anything", top_k=3, alpha=0.5)

        for r in results:
            assert not math.isnan(r["hybrid_score"]), "NaN in hybrid_score after Scenario C fix!"

    def test_result_has_score_breakdown(self):
        """Each result must expose score breakdown fields."""
        docs = self._make_docs()
        set_doc_store(docs)

        mock_bm25 = MagicMock()
        mock_bm25.query.return_value = [("doc_001", 2.0)]
        mock_vec = MagicMock()
        mock_vec.query.return_value = [("doc_001", 0.9)]

        with patch("app.search.hybrid.bm25_index", mock_bm25), \
             patch("app.search.hybrid.vector_index", mock_vec):
            results = hybrid_search("python", top_k=1)

        assert len(results) == 1
        r = results[0]
        for field in ["bm25_score", "bm25_score_norm", "vector_score", "vector_score_norm", "hybrid_score"]:
            assert field in r, f"Missing field: {field}"
