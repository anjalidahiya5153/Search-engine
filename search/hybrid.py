"""
Hybrid ranking: min-max normalise BM25 and semantic scores, then combine.

final_score = alpha * norm_bm25 + (1 - alpha) * norm_semantic
"""

from typing import Dict, List, Tuple

from config.settings import HYBRID_ALPHA, TOP_K
from search import bm25, semantic
from indexing.inverted_index import load_all_shards


def _min_max(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}

    lo, hi = min(scores.values()), max(scores.values())
    rng = hi - lo or 1.0

    return {
        k: (v - lo) / rng
        for k, v in scores.items()
    }


def search(
    query: str,
    top_k: int = TOP_K,
    alpha: float = HYBRID_ALPHA
) -> List[Tuple[str, float]]:
    """Return (doc_id, hybrid_score) sorted descending."""

    shards = load_all_shards()

    bm25_raw = dict(
        bm25.search(
            query,
            top_k=top_k * 2,
            shards=shards
        )
    )

    sem_raw = dict(
        semantic.search(
            query,
            top_k=top_k * 2
        )
    )

    bm25_norm = _min_max(bm25_raw)
    sem_norm = _min_max(sem_raw)

    all_ids = set(bm25_norm) | set(sem_norm)

    combined = {
        doc_id:
            alpha * bm25_norm.get(doc_id, 0.0)
            + (1 - alpha) * sem_norm.get(doc_id, 0.0)
        for doc_id in all_ids
    }

    return sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]