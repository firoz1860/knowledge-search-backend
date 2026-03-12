"""
Data ingestion pipeline.
Reads .txt / .md files from --input folder or sample corpus.
Normalizes to JSONL: doc_id, title, text, source, created_at.

Usage:
    python -m app.ingest --input data/raw --out data/processed
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as top-level module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import settings
from app.utils.preprocessing import clean_text, truncate_tokens
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAX_TEXT_TOKENS = 400  # safeguard for very long docs


def _make_doc_id(title: str, source: str) -> str:
    raw = f"{source}::{title}"
    return "doc_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def _parse_file(path: Path) -> list:
    """Parse a single .txt or .md file into one or more doc dicts."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Could not read {path}: {e}")
        return []

    # Split .md files on ## headings into sections
    if path.suffix == ".md":
        sections = re.split(r"^#{1,3}\s+", raw, flags=re.MULTILINE)
    else:
        sections = [raw]

    docs = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue
        lines = section.split("\n", 1)
        title = clean_text(lines[0][:120]) if lines else path.stem
        text = clean_text(lines[1] if len(lines) > 1 else section)
        text = truncate_tokens(text, max_tokens=MAX_TEXT_TOKENS)
        if len(text) < 30:
            continue
        doc_id = _make_doc_id(title, str(path.name))
        docs.append({
            "doc_id": doc_id,
            "title": title or path.stem,
            "text": text,
            "source": path.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return docs


def ingest(input_dir: str, out_dir: str) -> int:
    input_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory {input_path} does not exist")
        sys.exit(1)

    files = list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))
    if not files:
        logger.warning(f"No .txt or .md files found in {input_path}")
        return 0

    logger.info(f"Ingesting {len(files)} files from {input_path} …")
    all_docs = []
    seen_ids = set()
    for f in sorted(files):
        docs = _parse_file(f)
        for doc in docs:
            if doc["doc_id"] not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc["doc_id"])

    out_file = out_path / "docs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for doc in all_docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(all_docs)} documents → {out_file}")
    return len(all_docs)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into JSONL")
    parser.add_argument("--input", default=str(settings.raw_dir), help="Input directory with .txt/.md files")
    parser.add_argument("--out", default=str(settings.processed_dir), help="Output directory for docs.jsonl")
    args = parser.parse_args()
    count = ingest(args.input, args.out)
    print(f"✓ Ingested {count} documents")


if __name__ == "__main__":
    main()
