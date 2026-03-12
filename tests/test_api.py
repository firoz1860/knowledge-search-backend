"""
API contract tests using FastAPI TestClient.
Validates endpoint contracts without real indexes (mocked).
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# Minimal mocks so we can import app without real indexes
@pytest.fixture(autouse=True)
def mock_indexes():
    """Prevent actual index loading during tests."""
    with patch("app.search.bm25_search.bm25_index") as mock_bm25, \
         patch("app.search.vector_search.vector_index") as mock_vec:
        mock_bm25.is_loaded = True
        mock_bm25.num_docs = 5
        mock_bm25.exists.return_value = True
        mock_bm25.load.return_value = None
        mock_bm25.query.return_value = [("doc_001", 2.5), ("doc_002", 1.0)]

        mock_vec.is_loaded = True
        mock_vec.num_docs = 5
        mock_vec.exists.return_value = True
        mock_vec.load.return_value = None
        mock_vec.query.return_value = [("doc_001", 0.9), ("doc_002", 0.7)]

        yield mock_bm25, mock_vec


@pytest.fixture
def client(tmp_path, mock_indexes):
    # Point settings paths to temp dir
    with patch("app.config.settings.db_path", tmp_path / "test.db"), \
         patch("app.config.settings.processed_dir", tmp_path), \
         patch("app.config.settings.metrics_dir", tmp_path / "metrics"):

        from app.main import app
        from app.db import init_db
        with patch("app.main.init_db"):
            with TestClient(app) as c:
                yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "commit" in data
        assert "bm25_docs" in data
        assert "vector_docs" in data


class TestSearchEndpoint:
    def test_search_returns_200(self, client):
        with patch("app.api.routes.hybrid_search") as mock_search, \
             patch("app.api.routes.log_query"):
            mock_search.return_value = [{
                "doc_id": "doc_001", "title": "Test Doc", "snippet": "snippet",
                "source": "test.txt", "created_at": "2024-01-01T00:00:00Z",
                "bm25_score": 2.5, "bm25_score_norm": 1.0,
                "vector_score": 0.9, "vector_score_norm": 1.0,
                "hybrid_score": 1.0, "normalization": "minmax", "alpha": 0.5,
            }]
            resp = client.post("/api/search", json={"query": "python tutorial"})
        assert resp.status_code == 200

    def test_search_response_has_score_breakdown(self, client):
        with patch("app.api.routes.hybrid_search") as mock_search, \
             patch("app.api.routes.log_query"):
            mock_search.return_value = [{
                "doc_id": "doc_001", "title": "Test", "snippet": "...",
                "source": "src", "created_at": "2024-01-01",
                "bm25_score": 1.0, "bm25_score_norm": 0.8,
                "vector_score": 0.9, "vector_score_norm": 0.9,
                "hybrid_score": 0.85, "normalization": "minmax", "alpha": 0.5,
            }]
            resp = client.post("/api/search", json={"query": "test"})

        data = resp.json()
        assert "results" in data
        assert "latency_ms" in data
        assert "request_id" in data
        result = data["results"][0]
        for field in ["bm25_score", "vector_score", "hybrid_score", "bm25_score_norm", "vector_score_norm"]:
            assert field in result, f"Missing field: {field}"

    def test_search_validates_empty_query(self, client):
        resp = client.post("/api/search", json={"query": "   "})
        assert resp.status_code == 422

    def test_search_validates_alpha_range(self, client):
        resp = client.post("/api/search", json={"query": "test", "alpha": 1.5})
        assert resp.status_code == 422

    def test_search_validates_top_k_max(self, client):
        resp = client.post("/api/search", json={"query": "test", "top_k": 100})
        assert resp.status_code == 422

    def test_search_invalid_normalization(self, client):
        resp = client.post("/api/search", json={"query": "test", "normalization": "weird"})
        assert resp.status_code == 422


class TestFeedbackEndpoint:
    def test_feedback_accepted(self, client):
        with patch("app.api.routes.log_feedback"):
            resp = client.post("/api/feedback", json={
                "request_id": "req-123", "doc_id": "doc_001", "relevant": True,
            })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestMetricsEndpoint:
    def test_metrics_returns_text(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        assert "search_requests_total" in resp.text
        assert "latency_p50_ms" in resp.text


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        with patch("app.api.routes.get_query_stats") as mock_stats:
            mock_stats.return_value = {
                "total_requests": 10, "error_count": 0, "zero_result_count": 1,
                "p50_latency": 25.0, "p95_latency": 100.0,
                "top_queries": [], "zero_result_queries": [], "timeline": [],
            }
            resp = client.get("/api/stats")
        assert resp.status_code == 200
