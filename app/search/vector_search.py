"""
Vector search index: encode docs with sentence-transformers, store in FAISS (CPU).
Validates model/dimension on load to catch Scenario A (mismatch break/fix).
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss

from ..config import settings
from ..utils.logging_utils import get_logger
from ..utils.preprocessing import preprocess_for_indexing, truncate_tokens

logger = get_logger(__name__)

FAISS_INDEX_FILE = settings.vector_index_dir / "faiss.index"
VECTOR_META_FILE = settings.vector_index_dir / "meta.json"
VECTOR_DOCIDS_FILE = settings.vector_index_dir / "doc_ids.json"

# Encode batch size — keep small for CPU friendliness
ENCODE_BATCH = 64


class VectorIndex:
    def __init__(self):
        self._index: Optional[faiss.IndexFlatIP] = None  # inner-product (cosine after normalise)
        self._doc_ids: List[str] = []
        self._model = None
        self._model_name: str = ""
        self._dim: int = 0
        self.is_loaded: bool = False

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------
    def _load_model(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        if self._model is None or self._model_name != model_name:
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded — dim={self._dim}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, docs: List[Dict], model_name: Optional[str] = None) -> None:
        model_name = model_name or settings.embedding_model
        self._load_model(model_name)

        logger.info(f"Encoding {len(docs)} documents (batch={ENCODE_BATCH}) …")
        texts = [
            preprocess_for_indexing(d.get("title", ""), d.get("text", ""))
            for d in docs
        ]
        self._doc_ids = [d["doc_id"] for d in docs]

        # Encode in batches — show progress for large corpora
        all_vecs = []
        for i in range(0, len(texts), ENCODE_BATCH):
            batch = texts[i : i + ENCODE_BATCH]
            vecs = self._model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_vecs.append(vecs)
            if (i // ENCODE_BATCH) % 5 == 0:
                logger.info(f"  encoded {min(i + ENCODE_BATCH, len(texts))}/{len(texts)}")

        matrix = np.vstack(all_vecs).astype("float32")

        # Build FAISS IndexFlatIP (exact cosine, CPU)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)
        self._dim = matrix.shape[1]
        self.is_loaded = True
        logger.info(f"Vector index built: {self._index.ntotal} vectors, dim={self._dim}")

    # ------------------------------------------------------------------
    # Persist / Load
    # ------------------------------------------------------------------
    def save(self) -> None:
        settings.vector_index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(FAISS_INDEX_FILE))
        with open(VECTOR_DOCIDS_FILE, "w") as f:
            json.dump(self._doc_ids, f)
        meta = {
            "model_name": self._model_name,
            "dim": self._dim,
            "num_docs": len(self._doc_ids),
        }
        with open(VECTOR_META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Vector index saved to {settings.vector_index_dir}")

    def load(self) -> None:
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(f"Vector index not found at {FAISS_INDEX_FILE}. Run indexing first.")

        # Validate metadata before loading
        with open(VECTOR_META_FILE) as f:
            meta = json.load(f)

        stored_model = meta["model_name"]
        stored_dim = meta["dim"]
        current_model = settings.embedding_model

        # --- SCENARIO A: mismatch detection ---
        if stored_model != current_model:
            raise ValueError(
                f"Embedding model mismatch: index built with '{stored_model}' "
                f"but EMBEDDING_MODEL='{current_model}'. "
                "Delete data/index/vector/ and rebuild, or revert the model name."
            )

        self._index = faiss.read_index(str(FAISS_INDEX_FILE))
        actual_dim = self._index.d
        if actual_dim != stored_dim:
            raise ValueError(
                f"Dimension mismatch: meta says {stored_dim} but FAISS index has {actual_dim}. "
                "Index is corrupt — rebuild required."
            )

        with open(VECTOR_DOCIDS_FILE) as f:
            self._doc_ids = json.load(f)

        self._model_name = stored_model
        self._dim = actual_dim
        self._load_model(current_model)
        self.is_loaded = True
        logger.info(f"Vector index loaded: {len(self._doc_ids)} docs, dim={actual_dim}")

    def exists(self) -> bool:
        return FAISS_INDEX_FILE.exists() and VECTOR_META_FILE.exists()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return list of (doc_id, cosine_similarity) sorted descending.
        Scores are in [-1, 1]; higher is better.
        """
        if not self.is_loaded:
            raise RuntimeError("Vector index is not loaded. Call load() first.")
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        q_vec = self._model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
        q_vec = q_vec.astype("float32")

        k = min(top_k, len(self._doc_ids))
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx >= 0:
                results.append((self._doc_ids[idx], float(score)))
        return results

    @property
    def num_docs(self) -> int:
        return len(self._doc_ids)


# Module-level singleton
vector_index = VectorIndex()
