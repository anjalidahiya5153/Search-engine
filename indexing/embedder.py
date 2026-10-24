"""Sentence embedding using sentence-transformers (lazy-loaded singleton)."""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed(
    texts: List[str],
    batch_size: int = 64
) -> np.ndarray:
    """Return (N, EMBEDDING_DIM) float32 array."""

    model = _get_model()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return vectors.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]