"""
Retrieval quality metrics: NDCG@K, MRR, Recall@K.

All functions accept:
    retrieved: ordered list of doc_ids returned by the system
    relevant: set of relevant doc_ids (ground truth)
"""

import math
from typing import List, Set


def dcg(retrieved: List[str], relevant: Set[str], k: int) -> float:
    score = 0.0

    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / math.log2(rank + 1)

    return score


def ideal_dcg(relevant: Set[str], k: int) -> float:
    n = min(len(relevant), k)
    return sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, n + 1)
    )


def ndcg_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    idcg = ideal_dcg(relevant, k)

    if idcg == 0:
        return 0.0

    return dcg(retrieved, relevant, k) / idcg


def reciprocal_rank(
    retrieved: List[str],
    relevant: Set[str]
) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    if not relevant:
        return 0.0

    hits = sum(
        1 for d in retrieved[:k]
        if d in relevant
    )

    return hits / len(relevant)


def evaluate(
    queries: List[dict],
    search_fn,  # callable(query: str) -> List[str] (ordered doc_ids)
    k: int = 10,
) -> dict:
    """
    Evaluate a search function over a list of queries.

    Returns average NDCG@K, MRR, and Recall@K.
    """

    ndcgs, mrrs, recalls = [], [], []

    for item in queries:
        retrieved = search_fn(item["query"])
        relevant = set(item["relevant"])

        ndcgs.append(
            ndcg_at_k(retrieved, relevant, k)
        )

        mrrs.append(
            reciprocal_rank(retrieved, relevant)
        )

        recalls.append(
            recall_at_k(retrieved, relevant, k)
        )

    return {
        f"NDCG@{k}": sum(ndcgs) / len(ndcgs),
        "MRR": sum(mrrs) / len(mrrs),
        f"Recall@{k}": sum(recalls) / len(recalls),
        "num_queries": len(queries),
    }