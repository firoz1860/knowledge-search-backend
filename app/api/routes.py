"""
FastAPI route definitions.
Endpoints: GET /health, POST /search, POST /feedback, GET /metrics,
           GET /stats, GET /logs, GET /experiments
"""
import time
import uuid
import csv
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query
from fastapi.responses import PlainTextResponse

from .models import (
    SearchRequest, SearchResponse, SearchResult,
    FeedbackRequest, FeedbackResponse, HealthResponse,
)
from ..search import hybrid_search, bm25_index, vector_index
from ..db import log_query, log_feedback, get_query_stats, get_recent_logs
from ..config import settings
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

# In-memory counters for /metrics endpoint
_counters = {
    "search_total": 0,
    "search_errors": 0,
    "feedback_total": 0,
}
_latency_samples: list = []



@router.get("/health", response_model=HealthResponse)
async def health():
    db_ok = True
    try:
        from ..db import get_connection
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok",
        version=settings.version,
        commit=settings.git_commit,
        bm25_docs=bm25_index.num_docs,
        vector_docs=vector_index.num_docs,
        db_ok=db_ok,
    )



@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    error_msg: Optional[str] = None

    try:
        _counters["search_total"] += 1
        results_raw = hybrid_search(
            query=req.query,
            top_k=req.top_k,
            alpha=req.alpha,
            normalization=req.normalization,
            filters=req.filters,
        )
        results = [SearchResult(**r) for r in results_raw]

    except Exception as exc:
        _counters["search_errors"] += 1
        error_msg = str(exc)
        logger.error(f"Search error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        latency = (time.perf_counter() - start) * 1000
        _latency_samples.append(latency)
        if len(_latency_samples) > 1000:
            _latency_samples.pop(0)
        log_query(
            request_id=request_id,
            query=req.query,
            top_k=req.top_k,
            alpha=req.alpha,
            result_count=len(results) if error_msg is None else 0,
            latency_ms=latency,
            error=error_msg,
            filters=req.filters,
        )
        logger.info(
            "search_request",
            extra={
                "request_id": request_id, "query": req.query,
                "latency_ms": round(latency, 2), "top_k": req.top_k,
                "alpha": req.alpha, "result_count": len(results) if not error_msg else 0,
                "error": error_msg,
            },
        )

    return SearchResponse(
        request_id=request_id,
        query=req.query,
        results=results,
        result_count=len(results),
        latency_ms=round(latency, 2),
        alpha=req.alpha,
        normalization=req.normalization,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    _counters["feedback_total"] += 1
    log_feedback(req.request_id, req.doc_id, req.relevant)
    return FeedbackResponse()



@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    samples = _latency_samples
    p50 = _percentile(samples, 50)
    p95 = _percentile(samples, 95)
    lines = [
        "# HELP search_requests_total Total search requests",
        "# TYPE search_requests_total counter",
        f'search_requests_total {_counters["search_total"]}',
        "# HELP search_errors_total Total search errors",
        "# TYPE search_errors_total counter",
        f'search_errors_total {_counters["search_errors"]}',
        "# HELP feedback_total Total feedback events",
        "# TYPE feedback_total counter",
        f'feedback_total {_counters["feedback_total"]}',
        "# HELP latency_p50_ms p50 latency ms",
        "# TYPE latency_p50_ms gauge",
        f"latency_p50_ms {p50:.2f}",
        "# HELP latency_p95_ms p95 latency ms",
        "# TYPE latency_p95_ms gauge",
        f"latency_p95_ms {p95:.2f}",
        "# HELP bm25_index_docs BM25 index document count",
        f"bm25_index_docs {bm25_index.num_docs}",
        "# HELP vector_index_docs Vector index document count",
        f"vector_index_docs {vector_index.num_docs}",
    ]
    return "\n".join(lines)



@router.get("/stats")
async def stats(hours: int = Query(24, ge=1, le=168)):
    data = get_query_stats(hours=hours)
    # Attach live counters
    data["live_search_total"] = _counters["search_total"]
    data["live_error_total"] = _counters["search_errors"]
    return data


@router.get("/logs")
async def logs(
    limit: int = Query(100, ge=1, le=500),
    severity: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
):
    return get_recent_logs(limit=limit, severity=severity, hours=hours)



@router.get("/experiments")
async def experiments():
    csv_path = settings.metrics_dir / "experiments.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _percentile(samples: list, pct: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = int(len(s) * pct / 100)
    idx = min(idx, len(s) - 1)
    return s[idx]
