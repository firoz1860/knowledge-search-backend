"""
Text preprocessing utilities for ingest pipeline.
Handles whitespace normalisation, length safeguards, tokenisation.
"""
import re
import unicodedata
from typing import List


MAX_TOKENS = 512  # safeguard for extremely long docs


def normalize_whitespace(text: str) -> str:
    """Replace all whitespace sequences with a single space and strip."""
    if not text:
        return ""
    # Normalize unicode whitespace characters
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Remove control characters, normalise unicode, strip excess whitespace."""
    if not text:
        return ""
    # Remove control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = normalize_whitespace(text)
    return text


def truncate_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """
    Simple word-level truncation.
    sentence-transformers does its own sub-word truncation but we guard here too.
    """
    words = text.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens])
    return text


def simple_tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer: lowercase, remove punctuation, split on whitespace.
    Used by BM25 index so tokens must be consistent between build and query time.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    # Remove very short tokens (noise)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def extract_snippet(text: str, query: str, window: int = 150) -> str:
    """
    Return a short highlight snippet around the first occurrence of any query term.
    Falls back to the first `window` characters.
    """
    if not text or not query:
        return text[:window] + "…" if text else ""

    query_terms = [t.lower() for t in query.split() if len(t) > 2]
    text_lower = text.lower()

    best_pos = len(text)
    for term in query_terms:
        pos = text_lower.find(term)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(text):
        snippet = text[:window]
    else:
        start = max(0, best_pos - window // 3)
        end = min(len(text), start + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"

    # Bold the query terms (HTML-safe markers replaced in frontend)
    for term in query_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippet = pattern.sub(f"<mark>{term}</mark>", snippet)

    return snippet


def preprocess_for_indexing(title: str, text: str) -> str:
    """
    Combine title + text, clean and truncate.
    Used as the indexed content for both BM25 and vector search.
    """
    combined = f"{title} {title} {text}"  # double title for slight boost
    combined = clean_text(combined)
    combined = truncate_tokens(combined, max_tokens=MAX_TOKENS)
    return combined
