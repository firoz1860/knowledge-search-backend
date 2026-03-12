"""Tests for text preprocessing utilities."""
import pytest
from app.utils.preprocessing import (
    clean_text, normalize_whitespace, simple_tokenize,
    truncate_tokens, extract_snippet, preprocess_for_indexing,
)


class TestCleanText:
    def test_removes_control_chars(self):
        text = "hello\x00world\x07!"
        assert "\x00" not in clean_text(text)

    def test_normalizes_whitespace(self):
        result = clean_text("  hello   world  ")
        assert result == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_like_empty(self):
        assert clean_text("") == ""


class TestSimpleTokenize:
    def test_lowercases(self):
        tokens = simple_tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_punctuation(self):
        tokens = simple_tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_filters_short_tokens(self):
        tokens = simple_tokenize("a is an hello")
        assert "a" not in tokens
        assert "is" not in tokens  # len 2 — filtered

    def test_empty(self):
        assert simple_tokenize("") == []


class TestTruncateTokens:
    def test_does_not_truncate_short(self):
        text = "short text"
        assert truncate_tokens(text, max_tokens=512) == text

    def test_truncates_long(self):
        text = " ".join(["word"] * 1000)
        result = truncate_tokens(text, max_tokens=50)
        assert len(result.split()) == 50


class TestExtractSnippet:
    def test_contains_query_term(self):
        text = "The quick brown fox jumps over the lazy dog"
        snippet = extract_snippet(text, "fox", window=50)
        assert "fox" in snippet.lower()

    def test_fallback_on_no_match(self):
        text = "completely unrelated content here with extra words"
        snippet = extract_snippet(text, "zzz", window=30)
        # Should return beginning of text
        assert len(snippet) > 0

    def test_empty_text(self):
        assert extract_snippet("", "query") == ""

    def test_highlights_terms(self):
        text = "Python is a great programming language"
        snippet = extract_snippet(text, "python", window=100)
        assert "<mark>" in snippet


class TestPreprocessForIndexing:
    def test_title_doubled(self):
        result = preprocess_for_indexing("Python", "A programming language")
        # Title appears twice in combined text
        assert result.lower().count("python") >= 2

    def test_truncates(self):
        long_text = " ".join(["word"] * 1000)
        result = preprocess_for_indexing("title", long_text)
        assert len(result.split()) <= 515  # 512 + some title words
