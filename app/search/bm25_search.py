"""
BM25 index: build, persist, load, and query.
Uses rank-bm25's BM25Okapi which is CPU-only and dependency-free.
"""
import pickle
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi

from ..config import settings
from ..utils.logging_utils import get_logger
from ..utils.preprocessing import simple_tokenize, preprocess_for_indexing

logger = get_logger(__name__)

BM25_PICKLE = settings.bm25_index_dir / "bm25.pkl"
BM25_META = settings.bm25_index_dir / "meta.json"
BM25_DOCIDS = settings.bm25_index_dir / "doc_ids.json"


class BM25Index:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._corpus_hash: str = ""
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, docs: List[Dict]) -> None:
        """
        Build a BM25 index from a list of document dicts.
        Each doc must have: doc_id, title, text.
        """
        logger.info(f"Building BM25 index over {len(docs)} documents …")
        tokenized_corpus: List[List[str]] = []
        self._doc_ids = []

        for doc in docs:
            content = preprocess_for_indexing(doc.get("title", ""), doc.get("text", ""))
            tokens = simple_tokenize(content)
            tokenized_corpus.append(tokens)
            self._doc_ids.append(doc["doc_id"])

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._corpus_hash = self._hash_doc_ids(self._doc_ids)
        self.is_loaded = True
        logger.info("BM25 index built")

    # ------------------------------------------------------------------
    # Persist / Load
    # ------------------------------------------------------------------
    def save(self) -> None:
        settings.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        with open(BM25_PICKLE, "wb") as f:
            pickle.dump(self._bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(BM25_DOCIDS, "w") as f:
            json.dump(self._doc_ids, f)
        meta = {
            "corpus_hash": self._corpus_hash,
            "num_docs": len(self._doc_ids),
        }
        with open(BM25_META, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"BM25 index saved to {settings.bm25_index_dir}")

    def load(self) -> None:
        if not BM25_PICKLE.exists():
            raise FileNotFoundError(f"BM25 index not found at {BM25_PICKLE}. Run indexing first.")
        with open(BM25_PICKLE, "rb") as f:
            self._bm25 = pickle.load(f)
        with open(BM25_DOCIDS) as f:
            self._doc_ids = json.load(f)
        with open(BM25_META) as f:
            meta = json.load(f)
        self._corpus_hash = meta.get("corpus_hash", "")
        self.is_loaded = True
        logger.info(f"BM25 index loaded: {len(self._doc_ids)} docs")

    def exists(self) -> bool:
        return BM25_PICKLE.exists() and BM25_DOCIDS.exists()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return list of (doc_id, raw_bm25_score) sorted descending.
        Raw scores are non-negative floats; higher is better.
        """
        if not self.is_loaded:
            raise RuntimeError("BM25 index is not loaded. Call load() first.")
        tokens = simple_tokenize(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Pair with doc_ids and sort
        scored = sorted(
            zip(self._doc_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_doc_ids(doc_ids: List[str]) -> str:
        joined = ",".join(sorted(doc_ids))
        return hashlib.md5(joined.encode()).hexdigest()[:12]

    @property
    def num_docs(self) -> int:
        return len(self._doc_ids)


# Module-level singleton
bm25_index = BM25Index()
