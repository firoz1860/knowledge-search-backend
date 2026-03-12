"""
Evaluation harness: nDCG@10, Recall@10, MRR@10.
Appends results to data/metrics/experiments.csv.

Usage:
    python -m app.eval --queries data/eval/queries.jsonl --qrels data/eval/qrels.json
"""
import argparse
import csv
import json
import math
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import settings
from app.search.bm25_search import bm25_index
from app.search.vector_search import vector_index
from app.search.hybrid import hybrid_search, set_doc_store
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def dcg(relevances: list, k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            total += rel / math.log2(i + 2)
    return total


def ndcg(retrieved: list, relevant: set, k: int = 10) -> float:
    gains = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
    ideal = sorted(gains, reverse=True)
    actual = dcg(gains, k)
    perfect = dcg(ideal, k)
    return actual / perfect if perfect > 0 else 0.0


def recall_at_k(retrieved: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def mrr(retrieved: list, relevant: set, k: int = 10) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------

def run_eval(
    queries_path: str,
    qrels_path: str,
    alpha: float = 0.5,
    top_k: int = 10,
    normalization: str = "minmax",
    experiment_name: str = "",
) -> dict:
    # Load queries
    queries = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    # Load qrels: {query_id: [doc_id, ...]}
    with open(qrels_path) as f:
        qrels = json.load(f)

    logger.info(f"Running eval: {len(queries)} queries, alpha={alpha}, norm={normalization}")

    ndcg_scores, recall_scores, mrr_scores = [], [], []

    for q in queries:
        qid = q["query_id"]
        query_text = q["text"]
        relevant = set(qrels.get(qid, []))
        if not relevant:
            continue

        results = hybrid_search(
            query=query_text,
            top_k=top_k,
            alpha=alpha,
            normalization=normalization,
        )
        retrieved = [r["doc_id"] for r in results]

        ndcg_scores.append(ndcg(retrieved, relevant, k=top_k))
        recall_scores.append(recall_at_k(retrieved, relevant, k=top_k))
        mrr_scores.append(mrr(retrieved, relevant, k=top_k))

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name or f"alpha={alpha}_norm={normalization}",
        "alpha": alpha,
        "normalization": normalization,
        "top_k": top_k,
        "num_queries": len(ndcg_scores),
        "ndcg_at_10": avg(ndcg_scores),
        "recall_at_10": avg(recall_scores),
        "mrr_at_10": avg(mrr_scores),
        "git_commit": _git_commit(),
    }

    logger.info(f"Eval result: {result}")
    _append_to_csv(result)
    return result


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _append_to_csv(row: dict) -> None:
    path = settings.metrics_dir / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info(f"Appended eval result to {path}")


def _load_indexes():
    """Load indexes and doc store for standalone eval run."""
    docs_path = settings.processed_dir / "docs.jsonl"
    if not docs_path.exists():
        logger.error("Run ingest + index first")
        sys.exit(1)

    docs = []
    with open(docs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    bm25_index.load()
    vector_index.load()
    set_doc_store(docs)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--queries", default=str(settings.eval_dir / "queries.jsonl"))
    parser.add_argument("--qrels", default=str(settings.eval_dir / "qrels.json"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--normalization", default="minmax", choices=["minmax", "rank"])
    parser.add_argument("--experiment", default="", help="Experiment label")
    args = parser.parse_args()

    _load_indexes()
    result = run_eval(
        queries_path=args.queries,
        qrels_path=args.qrels,
        alpha=args.alpha,
        top_k=args.top_k,
        normalization=args.normalization,
        experiment_name=args.experiment,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
