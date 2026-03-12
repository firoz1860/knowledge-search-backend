"""
Pydantic request / response models for the API.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query string")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="BM25 weight (0=pure vector, 1=pure BM25)")
    normalization: str = Field("minmax", description="Score normalization: 'minmax' or 'rank'")
    filters: Optional[Dict[str, str]] = Field(None, description="Optional metadata filters")

    @field_validator("normalization")
    @classmethod
    def validate_normalization(cls, v: str) -> str:
        allowed = {"minmax", "rank"}
        if v not in allowed:
            raise ValueError(f"normalization must be one of {allowed}")
        return v

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace")
        return v


class SearchResult(BaseModel):
    doc_id: str
    title: str
    snippet: str
    source: str
    created_at: str
    bm25_score: float
    bm25_score_norm: float
    vector_score: float
    vector_score_norm: float
    hybrid_score: float
    normalization: str
    alpha: float


class SearchResponse(BaseModel):
    request_id: str
    query: str
    results: List[SearchResult]
    result_count: int
    latency_ms: float
    alpha: float
    normalization: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    relevant: bool


class FeedbackResponse(BaseModel):
    status: str = "ok"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    commit: str
    bm25_docs: int
    vector_docs: int
    db_ok: bool


class MetricsResponse(BaseModel):
    # Prometheus-compatible text format is returned directly from endpoint
    pass
