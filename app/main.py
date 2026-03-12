"""
FastAPI application entry point.
On startup: init DB, load indexes, populate doc store.
"""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .config import settings
from .db import init_db
from .api.routes import router
from .search import bm25_index, vector_index, set_doc_store
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_rpm}/minute"])

def _load_doc_store() -> list:
    docs_path = settings.processed_dir / "docs.jsonl"
    if not docs_path.exists():
        logger.warning(f"Processed docs not found at {docs_path}. Doc store empty.")
        return []

    docs = []
    with open(docs_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    logger.info(f"Loaded {len(docs)} docs into doc store")
    return docs
# def _load_doc_store() -> list:
#     docs_path = settings.processed_dir / "docs.jsonl"
#     if not docs_path.exists():
#         logger.warning(f"Processed docs not found at {docs_path}. Doc store empty.")
#         return []
#     docs = []
#     with open(docs_path) as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 docs.append(json.loads(line))
#     logger.info(f"Loaded {len(docs)} docs into doc store")
#     return docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, load indexes and doc store."""
    logger.info("=== Knowledge Search API starting up ===")
    settings.ensure_dirs()

    # Init DB
    init_db()

    # Load BM25 index
    try:
        if bm25_index.exists():
            bm25_index.load()
        else:
            logger.warning("BM25 index not found — search will return empty results until indexed.")
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")

    # Load vector index
    try:
        if vector_index.exists():
            vector_index.load()
        else:
            logger.warning("Vector index not found — search will return empty results until indexed.")
    except ValueError as e:
        # Scenario A: model mismatch — block startup with clear message
        logger.error(f"VECTOR INDEX MISMATCH: {e}")
        raise RuntimeError(f"Vector index validation failed: {e}")
    except Exception as e:
        logger.error(f"Failed to load vector index: {e}")

    # Populate doc store for snippets
    docs = _load_doc_store()
    set_doc_store(docs)

    logger.info("=== Startup complete ===")
    yield

    logger.info("=== Shutting down ===")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Hybrid Knowledge Search API — BM25 + Semantic vector search",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response

# Routes
app.include_router(router, prefix="/api")
