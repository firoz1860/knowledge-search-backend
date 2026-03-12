"""
Application configuration via environment variables / .env file.
"""
import os
from pathlib import Path

# Base project root (two levels up from this file: backend/app/ -> backend/ -> project/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    app_name: str = "Knowledge Search API"
    version: str = "1.0.0"
    git_commit: str = os.getenv("GIT_COMMIT", "dev")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Paths — all relative to project root so no hard-coded absolutes
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    index_dir: Path = BASE_DIR / "data" / "index"
    metrics_dir: Path = BASE_DIR / "data" / "metrics"
    eval_dir: Path = BASE_DIR / "data" / "eval"

    # Index sub-dirs
    bm25_index_dir: Path = BASE_DIR / "data" / "index" / "bm25"
    vector_index_dir: Path = BASE_DIR / "data" / "index" / "vector"

    # DB
    db_path: Path = BASE_DIR / "data" / "search.db"

    # Search defaults
    default_top_k: int = 10
    default_alpha: float = 0.5

    # Model — small CPU-friendly model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim: int = 384

    # Rate limiting (requests per minute per IP)
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "60"))

    # CORS origins
    cors_origins: list = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    def ensure_dirs(self):
        for d in [
            self.data_dir, self.raw_dir, self.processed_dir,
            self.index_dir, self.metrics_dir, self.eval_dir,
            self.bm25_index_dir, self.vector_index_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
