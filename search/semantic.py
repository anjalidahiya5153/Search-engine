"""Semantic search via FAISS."""
from typing import List, Tuple

from config.settings import TOP_K
from indexing.embedder import embed_one
from indexing import faiss_index


def search(
    query: str,
    top_k: int = TOP_K
) -> List[Tuple[str, float]]:
    """Return (doc_id, cosine similarity) sorted descending."""

    vec = embed_one(query)

    return faiss_index.search(
        vec,
        top_k
    )