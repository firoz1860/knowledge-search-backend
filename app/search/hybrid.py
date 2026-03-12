"""
Hybrid search: combine BM25 and vector scores with configurable alpha.
Supports two normalization strategies:
  1. min-max  — rescales each score to [0,1] over the retrieved set.
  2. rank      — converts positions to (1 / rank) scores (rank fusion).

Design choice documented in docs/decision_log.md.
Implements safe divide-by-zero handling to fix Scenario C.
"""
from typing import List, Dict, Tuple, Optional
import math

from .bm25_search import bm25_index
from .vector_search import vector_index
from ..utils.preprocessing import extract_snippet
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Doc store for snippets (populated at startup)
_doc_store: Dict[str, Dict] = {}


def set_doc_store(docs: List[Dict]) -> None:
    """Populate the in-memory doc store used for snippet extraction."""
    global _doc_store
    _doc_store = {d["doc_id"]: d for d in docs}
    logger.info(f"Doc store populated: {len(_doc_store)} docs")


def get_doc_store() -> Dict[str, Dict]:
    return _doc_store


# ------------------------------------------------------------------
# Normalisation helpers
# ------------------------------------------------------------------

def _minmax_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
    """
    Rescale scores to [0, 1].
    If all scores are identical (including all-zero), return 0.0 for every key
    rather than NaN. This is the FIX for Scenario C (divide-by-zero).
    """
    if not score_map:
        return {}
    lo = min(score_map.values())
    hi = max(score_map.values())
    span = hi - lo
    if span == 0:
        # All scores equal — return 0.0 uniformly (safe, no NaN)
        return {k: 0.0 for k in score_map}
    return {k: (v - lo) / span for k, v in score_map.items()}


def _rank_normalize(scored_list: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Reciprocal rank fusion: score = 1 / (k + rank) where k=60.
    Robust to outlier score distributions.
    """
    k = 60
    return {doc_id: 1.0 / (k + rank + 1) for rank, (doc_id, _) in enumerate(scored_list)}


# ------------------------------------------------------------------
# Hybrid search
# ------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    normalization: str = "minmax",  # "minmax" | "rank"
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """
    Perform hybrid retrieval and return ranked result dicts.

    hybrid_score = alpha * norm_bm25 + (1 - alpha) * norm_vector

    Returns:
        List of result dicts, sorted by hybrid_score desc, each containing:
        doc_id, title, text snippet, bm25_score, vector_score, hybrid_score, source.
    """
    # --- Fetch raw scores ---
    fetch_k = max(top_k * 3, 50)  # fetch more candidates before reranking

    bm25_raw = bm25_index.query(query, top_k=fetch_k)   # [(doc_id, score), ...]
    vec_raw  = vector_index.query(query, top_k=fetch_k) # [(doc_id, score), ...]

    bm25_map: Dict[str, float] = {doc_id: score for doc_id, score in bm25_raw}
    vec_map:  Dict[str, float] = {doc_id: score for doc_id, score in vec_raw}

    # Union of candidate doc_ids
    all_ids = set(bm25_map.keys()) | set(vec_map.keys())

    # Fill missing scores with 0.0
    for doc_id in all_ids:
        bm25_map.setdefault(doc_id, 0.0)
        vec_map.setdefault(doc_id, 0.0)

    # --- Normalise ---
    if normalization == "rank":
        bm25_norm = _rank_normalize(bm25_raw)
        vec_norm  = _rank_normalize(vec_raw)
        # Fill zeros for missing
        for doc_id in all_ids:
            bm25_norm.setdefault(doc_id, 0.0)
            vec_norm.setdefault(doc_id, 0.0)
    else:  # default: minmax
        bm25_norm = _minmax_normalize(bm25_map)
        vec_norm  = _minmax_normalize(vec_map)

    # --- Combine ---
    scored_results: List[Tuple[str, float, float, float]] = []
    for doc_id in all_ids:
        b = bm25_norm.get(doc_id, 0.0)
        v = vec_norm.get(doc_id, 0.0)
        hybrid = alpha * b + (1.0 - alpha) * v
        scored_results.append((doc_id, b, v, hybrid))

    # Sort by hybrid score descending
    scored_results.sort(key=lambda x: x[3], reverse=True)

    # --- Apply filters and build output ---
    output: List[Dict] = []
    for doc_id, b_norm, v_norm, h_score in scored_results:
        if len(output) >= top_k:
            break

        doc = _doc_store.get(doc_id)
        if doc is None:
            continue

        # Apply simple metadata filters
        if filters:
            if "source" in filters and doc.get("source") != filters["source"]:
                continue

        snippet = extract_snippet(doc.get("text", ""), query)

        output.append({
            "doc_id": doc_id,
            "title": doc.get("title", ""),
            "snippet": snippet,
            "source": doc.get("source", ""),
            "created_at": doc.get("created_at", ""),
            "bm25_score": round(bm25_map.get(doc_id, 0.0), 6),
            "bm25_score_norm": round(b_norm, 6),
            "vector_score": round(vec_map.get(doc_id, 0.0), 6),
            "vector_score_norm": round(v_norm, 6),
            "hybrid_score": round(h_score, 6),
            "normalization": normalization,
            "alpha": alpha,
        })

    logger.debug(
        "hybrid_search",
        extra={
            "query": query, "alpha": alpha, "top_k": top_k,
            "bm25_candidates": len(bm25_raw), "vec_candidates": len(vec_raw),
            "results": len(output),
        },
    )
    return output
