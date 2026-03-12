"""
Indexing pipeline: reads processed JSONL and builds BM25 + vector indexes.

Usage:
    python -m app.index_builder --input data/processed/docs.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import settings
from app.search.bm25_search import bm25_index
from app.search.vector_search import vector_index
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_docs(jsonl_path: str) -> list:
    path = Path(jsonl_path)
    if not path.exists():
        logger.error(f"Docs file not found: {path}")
        sys.exit(1)

    docs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    logger.info(f"Loaded {len(docs)} documents from {path}")
    return docs

# def load_docs(jsonl_path: str) -> list:
#     path = Path(jsonl_path)
#     if not path.exists():
#         logger.error(f"Docs file not found: {path}")
#         sys.exit(1)
#     docs = []
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 docs.append(json.loads(line))
#     logger.info(f"Loaded {len(docs)} documents from {path}")
#     return docs


def build_indexes(jsonl_path: str, model_name: str = None) -> None:
    docs = load_docs(jsonl_path)
    if not docs:
        logger.error("No documents to index")
        sys.exit(1)

    # BM25
    logger.info("--- Building BM25 index ---")
    bm25_index.build(docs)
    bm25_index.save()

    # Vector
    logger.info("--- Building vector index ---")
    vector_index.build(docs, model_name=model_name or settings.embedding_model)
    vector_index.save()

    logger.info("=== All indexes built successfully ===")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and vector indexes")
    parser.add_argument(
        "--input",
        default=str(settings.processed_dir / "docs.jsonl"),
        help="Path to processed docs.jsonl",
    )
    parser.add_argument(
        "--model",
        default=settings.embedding_model,
        help="Sentence-transformers model name",
    )
    args = parser.parse_args()
    build_indexes(args.input, args.model)
    print("✓ Indexes built")


if __name__ == "__main__":
    main()
