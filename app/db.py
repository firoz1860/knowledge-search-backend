"""
SQLite database layer: schema creation, migrations, query log persistence.
Supports simple versioned migrations to validate Scenario B (break/fix).
"""
import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

from .config import settings
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 2  # bump when schema changes


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1").fetchone()
        return row["version"] if row else 0
    except sqlite3.OperationalError:
        return 0


def migrate(conn: sqlite3.Connection):
    """Apply incremental migrations."""
    current = _get_current_version(conn)
    logger.info(f"DB schema version: {current}, target: {SCHEMA_VERSION}")

    if current < 1:
        logger.info("Applying migration v1: initial schema")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                query TEXT NOT NULL,
                top_k INTEGER NOT NULL,
                alpha REAL NOT NULL,
                result_count INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                relevant INTEGER NOT NULL CHECK(relevant IN (0,1)),
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at);
            CREATE INDEX IF NOT EXISTS idx_query_logs_query ON query_logs(query);
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(1)")
        conn.commit()

    if current < 2:
        logger.info("Applying migration v2: add filters column")
        conn.executescript("""
            ALTER TABLE query_logs ADD COLUMN filters TEXT;
        """)
        conn.execute("INSERT INTO schema_version(version) VALUES(2)")
        conn.commit()

    logger.info("DB migrations complete")


def init_db():
    """Initialize DB and run migrations."""
    settings.ensure_dirs()
    with db_conn() as conn:
        migrate(conn)
    logger.info(f"DB ready at {settings.db_path}")


def log_query(
    request_id: str,
    query: str,
    top_k: int,
    alpha: float,
    result_count: int,
    latency_ms: float,
    error: Optional[str] = None,
    filters: Optional[Dict] = None,
):
    try:
        with db_conn() as conn:
            conn.execute(
                """INSERT INTO query_logs
                   (request_id, query, top_k, alpha, result_count, latency_ms, error, filters)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (request_id, query, top_k, alpha, result_count, latency_ms, error,
                 json.dumps(filters) if filters else None),
            )
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


def log_feedback(request_id: str, doc_id: str, relevant: bool):
    try:
        with db_conn() as conn:
            conn.execute(
                "INSERT INTO feedback(request_id, doc_id, relevant) VALUES(?,?,?)",
                (request_id, doc_id, 1 if relevant else 0),
            )
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")


def get_query_stats(hours: int = 24) -> Dict[str, Any]:
    with db_conn() as conn:
        rows = conn.execute(
            """SELECT query, latency_ms, result_count, created_at, error
               FROM query_logs
               WHERE created_at >= datetime('now', ?)
               ORDER BY created_at DESC""",
            (f"-{hours} hours",),
        ).fetchall()

    if not rows:
        return {
            "total_requests": 0, "error_count": 0,
            "zero_result_count": 0, "p50_latency": 0, "p95_latency": 0,
            "top_queries": [], "zero_result_queries": [], "timeline": [],
        }

    latencies = sorted([r["latency_ms"] for r in rows])
    n = len(latencies)
    p50 = latencies[int(n * 0.5)] if n else 0
    p95 = latencies[min(int(n * 0.95), n - 1)] if n else 0

    from collections import Counter
    query_counts = Counter(r["query"] for r in rows)
    top_queries = [{"query": q, "count": c} for q, c in query_counts.most_common(10)]

    zero_result = [r["query"] for r in rows if r["result_count"] == 0]
    zero_counts = Counter(zero_result)
    zero_result_queries = [{"query": q, "count": c} for q, c in zero_counts.most_common(10)]

    # Timeline: requests per hour bucket
    from collections import defaultdict
    timeline_map: Dict[str, int] = defaultdict(int)
    for r in rows:
        hour_bucket = r["created_at"][:13]  # "YYYY-MM-DD HH"
        timeline_map[hour_bucket] += 1
    timeline = [{"time": k, "count": v} for k, v in sorted(timeline_map.items())]

    return {
        "total_requests": n,
        "error_count": sum(1 for r in rows if r["error"]),
        "zero_result_count": len(zero_result),
        "p50_latency": round(p50, 2),
        "p95_latency": round(p95, 2),
        "top_queries": top_queries,
        "zero_result_queries": zero_result_queries,
        "timeline": timeline,
    }


def get_recent_logs(limit: int = 100, severity: Optional[str] = None, hours: int = 24) -> List[Dict]:
    with db_conn() as conn:
        if severity == "error":
            rows = conn.execute(
                """SELECT * FROM query_logs
                   WHERE error IS NOT NULL AND created_at >= datetime('now',?)
                   ORDER BY created_at DESC LIMIT ?""",
                (f"-{hours} hours", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM query_logs
                   WHERE created_at >= datetime('now',?)
                   ORDER BY created_at DESC LIMIT ?""",
                (f"-{hours} hours", limit),
            ).fetchall()

    return [dict(r) for r in rows]
