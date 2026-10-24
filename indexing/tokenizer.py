"""Lightweight tokenizer used by BM25 and inverted index."""
import re
import string

# Common English stopwords (no external dependency)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that", "these",
    "those", "not", "no", "as", "if", "so", "he", "she", "we", "they", "you",
}

_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    return [t for t in tokens if len(t) > 1]